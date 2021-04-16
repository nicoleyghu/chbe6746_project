import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

ga_filename = "./ga_log_gen10_init25.csv"
hyperopt_filename = "./hyperopt_trials.txt"
nm_filename = "./neldermead_trials.txt"

df_ga = pd.read_csv(ga_filename, delimiter=",", header=None)
df_hyperopt = pd.read_csv(hyperopt_filename, delimiter="\t", header=None)
df_nm = pd.read_csv(nm_filename, delimiter="\t", header=None)

print(df_hyperopt.head(5))
print(df_nm.head(5))

ga_lr = df_ga.loc[:, 0].values.tolist()
ga_nodes = df_ga.loc[:, 2].values.tolist()
ga_layers = df_ga.loc[:, 1].values.tolist()
ga_mae = df_ga.loc[:, 3].values.tolist()

hyperopt_lr = df_hyperopt.loc[:, 0].values.tolist()
hyperopt_nodes = df_hyperopt.loc[:, 1].values.tolist()
hyperopt_layers = df_hyperopt.loc[:, 2].values.tolist()
hyperopt_mae = df_hyperopt.loc[:, 3].values.tolist()

nm_lr = df_nm.loc[:, 0].values.tolist()
nm_nodes = df_nm.loc[:, 1].values.tolist()
nm_layers = df_nm.loc[:, 2].values.tolist()
nm_mae = df_nm.loc[:, 3].values.tolist()

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 9))

ln01=axes[0].plot(range(0, len(hyperopt_lr)), hyperopt_lr, "--o",alpha=0.6, label="hyperopt")
ln02=axes[0].plot(range(0, len(nm_lr)), nm_lr, "--o",alpha=0.6, label="Nelder-Mead")
# axes[0].plot(range(0, len(ga_lr)), ga_lr, label="GA")
axes[0].set_yscale('log')

ax0_1 = axes[0].twiny()
ax0_1.get_shared_x_axes().join(axes[0], axes[1])
ax0_1.set_xlabel("num. of generations")
ln03=ax0_1.plot(range(0, len(ga_lr)), ga_lr, "g--o",alpha=0.6, label="GA")

lns = ln01 + ln02 + ln03
labs = [l.get_label() for l in lns]
axes[0].legend(lns, labs, loc=0)
axes[0].set_ylabel("lr")

ln01=axes[1].plot(range(0, len(hyperopt_nodes)), hyperopt_nodes, "--o",alpha=0.6, label="hyperopt")
ln02=axes[1].plot(range(0, len(nm_nodes)), nm_nodes, "--o",alpha=0.6, label="Nelder-Mead")
# axes[1].plot(range(0, len(ga_nodes)), ga_nodes, label="GA")

ax0_1 = axes[1].twiny()
ax0_1.get_shared_x_axes().join(axes[1], axes[0])
ln03=ax0_1.plot(range(0, len(ga_nodes)), ga_nodes, "g--o", alpha=0.6, label="GA")

ln01=axes[2].plot(range(0, len(hyperopt_layers)), hyperopt_layers, "--o",alpha=0.6, label="hyperopt")
ln02=axes[2].plot(range(0, len(nm_layers)), nm_layers, "--o",alpha=0.6, label="Nelder-Mead")
# axes[2].plot(range(0, len(ga_layers)), ga_layers, label="GA")
axes[1].set_ylabel("nodes")

ax0_1 = axes[2].twiny()
ax0_1.get_shared_x_axes().join(axes[2], axes[0])
ln03=ax0_1.plot(range(0, len(ga_layers)), ga_layers, "g--o", alpha=0.6, label="GA")

axes[2].set_ylabel("layers")

ln01=axes[3].plot(range(0, len(hyperopt_mae)), hyperopt_mae, "--o",alpha=0.6, label="hyperopt")
ln02=axes[3].plot(range(0, len(nm_mae)), nm_mae, "--o",alpha=0.6, label="Nelder-Mead")
# axes[3].plot(range(0, len(ga_mae)), ga_mae, label="GA")
axes[3].set_yscale('log')

ax0_1 = axes[3].twiny()
ax0_1.get_shared_x_axes().join(axes[3], axes[0])
ln03=ax0_1.plot(range(0, len(ga_mae)), ga_mae, "g--o",alpha=0.6, label="GA")

axes[3].set_xlabel("num. of trials")
axes[3].set_ylabel("mae")


for ax in axes:
    ax.get_yaxis().set_label_coords(-0.1,0.5)

fig.savefig("./figures/p2_results.pdf")

