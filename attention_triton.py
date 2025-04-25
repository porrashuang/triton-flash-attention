import triton
import triton.language as tl
import torch, math

# Implementation according to:
# https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
@triton.jit
def flashatt_onehead(Q_ptr, K_ptr, V_ptr, O_ptr,
                     seq_len: tl.constexpr,
                     d: tl.constexpr,             # head dim, e.g. 64
                     BLOCK_M: tl.constexpr = 64,
                     BLOCK_N: tl.constexpr = 64):

    pid = tl.program_id(0)             # which query tile
    q_row = pid * BLOCK_M
    offs_m = q_row + tl.arange(0, BLOCK_M)          # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                  # [BLOCK_N]
    offs_d = tl.arange(0, d)                        # [d]

    # ---- load Q tile (BLOCK_M × d) ----
    q = tl.load(Q_ptr + (offs_m[:, None] * d + offs_d[None, :]))

    # running softmax state
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32) # A tile of matrix A
    # running max state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    # running sum state of exponentials
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # ---- loop over K/V tiles ----
    for k_row in range(0, seq_len, BLOCK_N):
        # load K tile (BLOCK_N × d) and transpose, now (d × BLOCK_N)
        k = tl.load(K_ptr + ((k_row + offs_n)[:, None] * d + offs_d[None, :]))
        k = tl.trans(k)
        # score is a matrix (BLOCK_M × BLOCK_N)
        scores = tl.dot(q, k) / math.sqrt(d)

        # row-wise maximum, find max in BLOCK_N
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))

        # exponential power in old scale, the first term when update d_i
        # Shape of p_old_scale is (BLOCK_M, 1)
        p_old_scale = tl.exp(m_i - m_i_new)
        # the second term when update d_i
        # Shape of scores_shift is (BLOCK_M, BLOCK_N)
        scores_shift = scores - m_i_new[:, None]
        # Shape of scores_shift_sum is (BLOCK_M,)
        scores_shift_sum = tl.sum(tl.exp(scores_shift), axis=1)
        d_i_new = (d_i * p_old_scale + scores_shift_sum)

        # load V tile (BLOCK_N × d)
        v = tl.load(V_ptr + ((k_row + offs_n)[:, None] * d + offs_d[None, :]))
        # shape of v_scores_shift is (BLOCK_M, d)
        v_scores_shift_sum = tl.sum(tl.exp(scores_shift)[:, :, None] / d_i_new[:, None, None] * v[None, :, :], axis=1)
        acc_old_scale = acc * d_i[:, None] * p_old_scale[:, None] / d_i_new[:, None]

        m_i = m_i_new
        acc = acc_old_scale + v_scores_shift_sum
        d_i = d_i_new

    # ---- final normalisation & store ----
    # Shape of out is (BLOCK_M, d)
    tl.store(O_ptr + (offs_m[:, None] * d + offs_d[None, :]), acc)

def attention_triton(Q, K, V):
    """
    Triton implementation of the attention mechanism.
    """
    seq_len, input_dim = Q.shape
    BLOCK_M, BLOCK_N = 64, 64               # tune for your GPU
    grid = (triton.cdiv(seq_len, BLOCK_M),) # 1d grid for batch dimension

    O = torch.empty_like(Q, dtype=Q.dtype)

    flashatt_onehead[grid](Q, K, V, O,
                           seq_len=seq_len, d=input_dim,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return O

