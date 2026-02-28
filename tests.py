import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hjortMatrix import Matrix as M

sizes = [100, 500, 1000, 2000, 5000]
iterations = 15 
single_avg = []
multi_avg = []
numpy_avg = []

total_steps = len(sizes) * iterations
current_step = 0

def update_progress(current, total, size, iteration):
    percent = (current / total) * 100
    bar_length = 30
    done = int(bar_length * current / total)
    bar = "█" * done + "-" * (bar_length - done)
    sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Size: {size}x{size} | Iter: {iteration}/{iterations}")
    sys.stdout.flush()

for n in sizes:
    s_times, m_times, n_times = [], [], []
    
    for i in range(1, iterations + 1):
        current_step += 1
        update_progress(current_step, total_steps, n, i)

        A_st = M.random(n, n, multithreaded=False)
        B_st = M.random(n, n, multithreaded=False)
        A_mt = M.random(n, n, multithreaded=True)
        B_mt = M.random(n, n, multithreaded=True)
        A_np = np.random.rand(n, n)
        B_np = np.random.rand(n, n)

        start = time.perf_counter()
        C_st = A_st + B_st
        s_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        C_mt = A_mt + B_mt
        m_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        C_np = A_np + B_np
        n_times.append(time.perf_counter() - start)

        del A_st, B_st, C_st, A_mt, B_mt, C_mt, A_np, B_np, C_np

    single_avg.append(np.mean(s_times))
    multi_avg.append(np.mean(m_times))
    numpy_avg.append(np.mean(n_times))

print("\nTests complete.")

plt.figure(figsize=(10, 6))
plt.plot(sizes, single_avg, label="HJORT Single-threaded", marker='o')
plt.plot(sizes, multi_avg, label="HJORT Multithreaded", marker='s')
plt.plot(sizes, numpy_avg, label="NumPy", marker='^')
plt.xlabel("Matrix size (N x N)")
plt.ylabel("Time (seconds)")
plt.title("Matrix Addition Performance: Single vs Multi-threaded vs NumPy")
plt.legend()
plt.grid(True)
plt.show()