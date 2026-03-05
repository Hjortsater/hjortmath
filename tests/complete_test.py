import time
import numpy as np
import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('TkAgg')
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hjortMatrix import Matrix as M

# -----------------------------
# Settings
# -----------------------------
# Slightly adjusted sizes for better balance between speed and data
sizes = list(range(500, 3500, 500))
runs_per_size = 3
np.random.seed(42)

plt.style.use("dark_background")

# -----------------------------
# Outlier Removal (IQR method)
# -----------------------------
def remove_outliers(data):
    if len(data) < 3: return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]

# -----------------------------
# Benchmark Function
# -----------------------------
def benchmark(operation_name):
    hjort_raw = []
    numpy_raw = []

    for n in sizes:
        print(f"{operation_name.upper()}: {n}x{n}")

        # Pre-generate data to avoid timing generation
        A_data = np.random.rand(n, n)
        B_data = np.random.rand(n, n)

        hjort_runs = []
        numpy_runs = []

        for _ in range(runs_per_size):
            # Hjort: Use from_buffer if available for speed, else stay with your constructor
            # A = M.from_buffer(A_data) is much faster than A = M(*A_data.tolist())
            A = M.from_buffer(A_data) if hasattr(M, 'from_buffer') else M(*A_data.tolist())
            B = M.from_buffer(B_data) if hasattr(M, 'from_buffer') else M(*B_data.tolist())
            
            start = time.perf_counter()
            if operation_name == "add":
                _ = A + B
            elif operation_name == "sub":
                _ = A - B
            elif operation_name == "mul":
                _ = A * B
            elif operation_name == "inv":
                _ = A.inverse
            hjort_runs.append(time.perf_counter() - start)

            # NumPy
            start = time.perf_counter()
            if operation_name == "add":
                _ = A_data + B_data
            elif operation_name == "sub":
                _ = A_data - B_data
            elif operation_name == "mul":
                _ = np.dot(A_data, B_data)
            elif operation_name == "inv":
                _ = np.linalg.inv(A_data)
            numpy_runs.append(time.perf_counter() - start)

        # Remove outliers per size
        hjort_raw.append(remove_outliers(np.array(hjort_runs)))
        numpy_raw.append(remove_outliers(np.array(numpy_runs)))

    return hjort_raw, numpy_raw

# -----------------------------
# Plotting helper
# -----------------------------
def plot_with_fit(ax, sizes, raw_data, label, color, degree):
    means = np.array([arr.mean() for arr in raw_data])
    stds = np.array([arr.std() for arr in raw_data])

    # Polynomial fit
    coeffs = np.polyfit(sizes, means, degree)
    poly = np.poly1d(coeffs)
    smooth_x = np.linspace(min(sizes), max(sizes), 200)

    # Raw points
    for i, size in enumerate(sizes):
        ax.scatter([size] * len(raw_data[i]), raw_data[i], color=color, alpha=0.2, s=15)

    ax.plot(sizes, means, 'o', color=color, label=f"{label} mean")
    ax.plot(smooth_x, poly(smooth_x), '-', color=color, linewidth=2)
    ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.1)

# -----------------------------
# Run benchmarks
# -----------------------------
add_hjort, add_numpy = benchmark("add")
mul_hjort, mul_numpy = benchmark("mul")
inv_hjort, inv_numpy = benchmark("inv")

# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(21, 6))

# Addition
plot_with_fit(axs[0], sizes, add_hjort, "HjortMatrix", "#00BFFF", degree=2)
plot_with_fit(axs[0], sizes, add_numpy, "NumPy", "#FF4C4C", degree=2)
axs[0].set_title("Addition (O(n²))", fontsize=12)
axs[0].set_ylabel("Time (seconds)")
axs[0].legend()

# Multiplication
plot_with_fit(axs[1], sizes, mul_hjort, "HjortMatrix", "#00BFFF", degree=3)
plot_with_fit(axs[1], sizes, mul_numpy, "NumPy", "#FF4C4C", degree=3)
axs[1].set_title("Multiplication (O(n³))", fontsize=12)
axs[1].legend()

# Inversion
plot_with_fit(axs[2], sizes, inv_hjort, "HjortMatrix", "#00BFFF", degree=3)
plot_with_fit(axs[2], sizes, inv_numpy, "NumPy", "#FF4C4C", degree=3)
axs[2].set_title("Inversion (O(n³))", fontsize=12)
axs[2].legend()

for ax in axs:
    ax.set_xlabel("Matrix Size (n x n)")
    ax.grid(alpha=0.2)

plt.suptitle(f"Matrix Benchmark: Hjort vs NumPy\n(Sizes {sizes[0]}-{sizes[-1]}, {runs_per_size} Runs/Size)", fontsize=14)
plt.tight_layout()
plt.show()