import numpy as np
import glob
import os
import matplotlib.pyplot as plt

results = {}

for path in sorted(glob.glob("plot results/*npz")):
    data = np.load(path)
    label = os.path.basename(path).removesuffix(".npz")
    results[label] = {
        "times": data["stats_array"][:, 3],
        "healthy": data["stats_array"][:, 0],
        "infected": data["stats_array"][:, 1],
        "dead": data["stats_array"][:, 2],
        "MOI": float(data["MOI"]),
        "time_delay": float(data["time_delay"])
    }
    


plot_title = f"M_I = 7"

fig, ax = plt.subplots()
for label, r in results.items():
    ax.plot(r["times"], r["dead"], label = label)

ax.set(xlabel = "Time [h]", ylabel = "Dead cells", title = plot_title)
ax.legend()
ax.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("M_I=7.png")
plt.show()
