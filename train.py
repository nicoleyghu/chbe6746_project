import numpy as np
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch

from ase.visualize import view

images = read("./data/water_dft.traj", index=":")

sigmas = np.logspace(np.log10(0.02), np.log10(1.0), num=5)
MCSHs = {
    "MCSHs": {
        "0": {"groups": [1], "sigmas": sigmas},
        "1": {"groups": [1], "sigmas": sigmas},
        "2": {"groups": [1, 2], "sigmas": sigmas},
        "3": {"groups": [1, 2, 3], "sigmas": sigmas},
        "4": {"groups": [1, 2, 3, 4], "sigmas": sigmas},
        "5": {"groups": [1, 2, 3, 4, 5], "sigmas": sigmas},
        # "6": {"groups": [1, 2, 3, 4, 5, 6, 7], "sigmas": sigmas},
    },
    "atom_gaussians": {
        "H": "./MCSH_potentials/H_pseudodensity_2.g",
        "O": "./MCSH_potentials/O_pseudodensity_4.g",
    },
    "cutoff": 8,
}


elements = ["H", "O"]
config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.2,
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 500,
    },
    "dataset": {
        "raw_data": images,
        # "val_split": 0.1,
        "elements": elements,
        "fp_scheme": "gmp",
        "fp_params": MCSHs,
        "save_fps": True,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        "logger": False,
    },
}

trainer = AtomsTrainer(config)
trainer.train()

predictions = trainer.predict(images)

true_energies = np.array([image.get_potential_energy() for image in images])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))
