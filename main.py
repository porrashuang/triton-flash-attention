import torch
import triton
from attention_pytorch import attention_pytorch
from attention_triton import attention_triton

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
configs = []
input_dim = 64
dtype = torch.float32
seq_lens = [2**i for i in range(6, 19)]
configs.append(
    triton.testing.Benchmark(
        line_arg="provider",
        line_vals=['pytorch', 'triton'],
        line_names=['pytorch', 'triton'],
        x_names=["seq_len"],
        x_log=True,
        x_vals=seq_lens,
        ylabel='time (ms)',
        y_log=True,
        plot_name='attention_performance',
        args=
        {
            "input_dim": input_dim,
        }
    )
)

pytorch_memory = {key: [] for key in seq_lens}
triton_memory = {key: [] for key in seq_lens}
# Wrapper for attnetion_pytorch and attention_triton to record memory usage
def attention_pytorch_wrapped(seq_len, Q, K, V):
    torch.cuda.reset_peak_memory_stats()
    try:
        O = attention_pytorch(Q, K, V)
    except torch.cuda.OutOfMemoryError:
        print("Out of memory error in PyTorch attention.")
        O = None
    pytorch_memory[seq_len].append(torch.cuda.max_memory_allocated() / 1e6)  # Convert to MB
    return O

def attention_triton_wrapped(seq_len, Q, K, V):
    torch.cuda.reset_peak_memory_stats()
    try:
        O = attention_triton(Q, K, V)
    except torch.cuda.OutOfMemoryError:
        print("Out of memory error in Triton attention.")
        O = None
    triton_memory[seq_len].append(torch.cuda.max_memory_allocated() / 1e6)  # Convert to MB
    return O
    

@triton.testing.perf_report(configs)
def benchmark_attention(seq_len, input_dim , provider, device=DEVICE):

    # Generate random Q, K, V
    Q = torch.randn(seq_len, input_dim, device=device, dtype=dtype)
    K = torch.randn(seq_len, input_dim, device=device, dtype=dtype)
    V = torch.randn(seq_len, input_dim, device=device, dtype=dtype)

    if provider == 'pytorch':
        # Measure PyTorch attention
        fn = lambda: attention_pytorch_wrapped(seq_len, Q, K, V)
        ms = triton.testing.do_bench(fn, warmup=2, rep=5)
    elif provider == 'triton':
        # Measure Triton attention
        fn = lambda: attention_triton_wrapped(seq_len, Q, K, V)
        ms = triton.testing.do_bench(fn, warmup=2, rep=5)

    # total_flops = 2 * seq_len * input_dim * input_dim
    # return total_flops / ms * 1e-6  # Convert to GFLOPS
    return ms

# Plotting memory usage
def plot_memory():
    import matplotlib.pyplot as plt
    import numpy as np
    seq_lens = sorted(pytorch_memory.keys())
    pytorch_mem = [np.mean(pytorch_memory[seq_len]) for seq_len in seq_lens]
    triton_mem = [np.mean(triton_memory[seq_len]) for seq_len in seq_lens]
    plt.figure(figsize=(10, 5))
    plt.plot(seq_lens, pytorch_mem, label='PyTorch', marker='o')
    plt.plot(seq_lens, triton_mem, label='Triton', marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    # Use log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Memory Usage of Attention Mechanism')
    plt.legend()
    plt.grid()
    plt.savefig('results/memory_usage.png')
    plt.show()
    # Save memory usage data
    with open('results/memory_usage.txt', 'w') as f:
        f.write("Sequence Length\tPyTorch Memory (MB)\tTriton Memory (MB)\n")
        for seq_len in seq_lens:
            f.write(f"{seq_len}\t{np.mean(pytorch_memory[seq_len]):.2f}\t{np.mean(triton_memory[seq_len]):.2f}\n")


def unit_test_attention():
    # Test the Triton implementation
    seq_len = 64
    Q = torch.randn(seq_len, input_dim, device=DEVICE, dtype=dtype)
    K = torch.randn(seq_len, input_dim, device=DEVICE, dtype=dtype)
    V = torch.randn(seq_len, input_dim, device=DEVICE, dtype=dtype)

    O_triton = attention_triton(Q, K, V)
    O_pytorch = attention_pytorch(Q, K, V)

    if not torch.allclose(O_triton, O_pytorch, atol=1e-2):
        print("Triton Output:")
        print(O_triton)
        print("PyTorch Output:")
        print(O_pytorch)
        raise AssertionError("Triton and PyTorch outputs do not match!")
    else:
        print("Unit test passed!")

    
def main():
   unit_test_attention()
   benchmark_attention.run(save_path='results/', print_data=True, show_plots=True)
   plot_memory()

if __name__ == "__main__":
    main()