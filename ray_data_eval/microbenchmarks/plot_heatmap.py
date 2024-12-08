import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, Normalize, to_rgba

# Data
x_axis = [4, 8, 12, 16]
y_axis_labels = ['Radar', 'Flink', 'Spark', 'TFData']

BIG_SIZE   = 9
FIGRATIO = 3 / 4
FIGWIDTH = 3.335 # inches
FIGHEIGHT = FIGWIDTH * FIGRATIO
FIGSIZE = (FIGWIDTH, FIGHEIGHT)

plt.rcParams.update(
{
    "figure.figsize": FIGSIZE,
    "figure.dpi": 300,
    # "text.usetex": True,
}
)

COLORS = sns.color_palette("Paired")
sns.set_style("ticks")
sns.set_palette(COLORS)

plt.rc("font", size=BIG_SIZE) # controls default text sizes
plt.rc("axes", titlesize=BIG_SIZE) # fontsize of the axes title
plt.rc("axes", labelsize=BIG_SIZE) # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIG_SIZE) # fontsize of the tick labels
plt.rc("ytick", labelsize=BIG_SIZE) # fontsize of the tick labels
plt.rc("legend", fontsize=BIG_SIZE) # legend fontsize
plt.rc("figure", titlesize=BIG_SIZE) # fontsize of the figure title

data = [
    [np.nan, 200, 200, 200],        # Radar
    [np.nan, np.nan, 429, 428],  # Flink
    [np.nan, np.nan, 487, 204], # Spark
    [np.nan, 520, 520, 449]      # TFData
]

# Create a DataFrame
df = pd.DataFrame(data, index=y_axis_labels, columns=x_axis)

# Define NaN color and gradient colormap
nan_color = to_rgba("grey")  # Dark brown for NaN
data_cmap = sns.color_palette("RdYlGn_r", as_cmap=True)  # Red-to-green gradient for numeric values

# Build a custom colormap
gradient_colors = data_cmap(np.linspace(0, 1, 255))  # Gradient for valid data
custom_cmap = ListedColormap(list(gradient_colors))  # Add NaN color as the first color

# Mask for NaN values
# nan_mask = df.isnull()

nan_mask = df.isnull()



# Plot
ax = sns.heatmap(df, annot=True, linewidths=1, linecolor='white', fmt=".0f", cmap=custom_cmap, cbar_kws={'label': 'JCT (s)'})

plt.imshow(nan_mask, cmap=ListedColormap([nan_color]), alpha=0.4, zorder=-10, extent=ax.get_xlim() + ax.get_ylim())

# Overlay NaN regions with the distinct color
# plt.imshow(nan_mask, cmap=ListedColormap([nan_color]), alpha=0.7)

# Axis labels and title
plt.xlabel("Memory (GB)")
plt.ylabel("Systems")
# plt.title("Heatmap with Proper Scaling and NaN Highlighted as Dark Brown")

# Save and display the plot
plt.tight_layout()
plt.savefig("heatmap_corrected.pdf")
plt.show()