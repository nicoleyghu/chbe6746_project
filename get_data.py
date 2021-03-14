import numpy as np
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch

import pickle


images = read("./data/water.traj", index=":")

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
    "model": {"get_forces": False, "num_layers": 3, "num_nodes": 20},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.0,
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 2,
    },
    "dataset": {
        "raw_data": images,
        # "val_split": 0.1,
        "elements": elements,
        "fp_scheme": "mcsh",
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

dataset = trainer.train_dataset

position_array = []
fp_array = []
energy_array = []


for i, data in enumerate(dataset):
    position = images[i].get_positions()
    position_array.append(np.asarray(position))
    fingerprint = data.fingerprint.numpy()
    fp_array.append(fingerprint)
    energy = images[i].get_potential_energy()
    energy_array.append(energy)

positions_pkl = "positions.pkl"
fingerprints_pkl = "fingerprints.pkl"
energies_pkl = "energies.pkl"

with open(positions_pkl, "wb") as f:
    pickle.dump(np.asarray(position_array), f)

with open(fingerprints_pkl, "wb") as f:
    pickle.dump(np.asarray(fp_array), f)

with open(energies_pkl, "wb") as f:
    pickle.dump(np.asarray(energy_array), f)

