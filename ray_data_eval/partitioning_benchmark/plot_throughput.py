import matplotlib.pyplot as plt
import seaborn as sns
import re

# Style settings
FIGRATIO = 3 / 4
FIGWIDTH = 4
FIGHEIGHT = FIGWIDTH * FIGRATIO
FIGSIZE = (FIGWIDTH, FIGHEIGHT)
BIG_SIZE = 14

COLORS = sns.color_palette("Paired")
sns.set_style("ticks")
sns.set_palette(COLORS)

plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "figure.dpi": 300,
        "text.usetex": True,
    }
)
plt.rc("font", size=BIG_SIZE)
plt.rc("axes", titlesize=BIG_SIZE)
plt.rc("axes", labelsize=BIG_SIZE)
plt.rc("xtick", labelsize=BIG_SIZE)
plt.rc("ytick", labelsize=BIG_SIZE)
plt.rc("legend", fontsize=BIG_SIZE)
plt.rc("figure", titlesize=BIG_SIZE)

# Data processing
DATA = """
num_rows_in_block=1, duration=84.5572521686554
num_rows_in_block=2, duration=62.60961127281189
num_rows_in_block=4, duration=52.042343854904175
num_rows_in_block=8, duration=36.00739049911499
num_rows_in_block=16, duration=32.76410746574402
num_rows_in_block=32, duration=30.122849702835083
num_rows_in_block=64, duration=29.180481433868408
num_rows_in_block=128, duration=29.577149391174316
num_rows_in_block=256, duration=31.33970308303833
num_rows_in_block=512, duration=34.02229189872742
num_rows_in_block=1024, duration=49.8162796497345
"""

# Extract data using regex
num_rows = [
    int(re.search(r"num_rows_in_block=(\d+)", line).group(1)) for line in DATA.strip().split("\n")
]
durations = [
    float(re.search(r"duration=(\d+\.?\d*)", line).group(1)) for line in DATA.strip().split("\n")
]

# Calculate throughput
throughput = [8192 / duration for duration in durations]

# Create the plot with logarithmic x-axis
plt.figure()
ax = plt.gca()
plt.bar(num_rows, throughput, width=[n * 0.5 for n in num_rows])  # Width proportional to x value

ax.set_xscale("log", base=2)  # Use log scale with base 2
exps = [0, 3, 5, 7, 10]
tick_positions = [2**n for n in exps]
tick_labels = ["$2^{" + str(n) + "}$" for n in exps]  # LaTeX formatting
tick_labels[0] = "1"  # Replace first label with 1
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

plt.xlabel("Partition size (MB)")
plt.ylabel("Throughput (rows/s)")
plt.grid(True, which="both", ls="-", alpha=0.5)  # Add grid for better readability

# Save the plot
plt.savefig("partition-throughput.pdf", bbox_inches="tight")
plt.savefig("partition-throughput.png", bbox_inches="tight")
