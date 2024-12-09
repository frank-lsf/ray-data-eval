import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

FIGRATIO = 3 / 5
FIGWIDTH = 5  # inches
FIGHEIGHT = FIGWIDTH * FIGRATIO
FIGSIZE = (FIGWIDTH, FIGHEIGHT)

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "figure.figsize": FIGSIZE,
        "figure.dpi": 300,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": True,
    }
)

COLORS = sns.color_palette("Paired")
sns.set_style("ticks")
sns.set_palette(COLORS)

# Read both CSV files
df_dynamic = pd.read_csv("../raydata/dynamic_tput.csv")
df_static = pd.read_csv("../raydata/static_tput.csv")

# Process dynamic data
x_dynamic = df_dynamic["time_from_start"]
y_dynamic = df_dynamic["batch_throughput"]
X_smooth_dynamic = np.linspace(x_dynamic.min(), x_dynamic.max(), 300)
Y_smooth_dynamic = np.interp(X_smooth_dynamic, x_dynamic, y_dynamic)

# Process static data
x_static = df_static["time_from_start"]
y_static = df_static["batch_throughput"]
X_smooth_static = np.linspace(x_static.min(), x_static.max(), 300)
Y_smooth_static = np.interp(X_smooth_static, x_static, y_static)

# Apply smoothing to both series
window_size = 10
kernel = np.ones(window_size) / window_size
Y_smooth_dynamic = np.convolve(Y_smooth_dynamic, kernel, mode="same")
Y_smooth_static = np.convolve(Y_smooth_static, kernel, mode="same")

# Create the plot
plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
plt.plot(X_smooth_dynamic, Y_smooth_dynamic, "b-", label="Radar (Adaptive)")
plt.plot(X_smooth_static, Y_smooth_static, "r-", label="Radar (Static)")

plt.xlabel("Time (s)")
plt.ylabel("Throughput (frames/s)")
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.savefig("video_tput.pdf")
plt.savefig("video_tput.png")
