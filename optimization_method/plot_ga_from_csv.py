import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

# read csv
ga_log = "./ga_log.csv"

ga_results = pd.read_csv(ga_log)

fig, ax = plt.subplots()
ax.plot(ga_results.iloc[:,1], ga_results.iloc[:,0], "-o")
ax.set_xlabel("bend")
ax.set_ylabel("stretch")
ax.set_title("GA: Ensemble 0")

fig.savefig("ga_result.png", dpi=200)
