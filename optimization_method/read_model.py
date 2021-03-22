import numpy as np
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.optimize import BFGS

from ase.build import molecule

def calculate_energy(x0):
    OH_bond_length = x0[0] 
    bond_angle = x0[1]
    images = read("../data/water.traj", index=":")
    images = [images[0]]

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
            "H": "../MCSH_potentials/H_pseudodensity_2.g",
            "O": "../MCSH_potentials/O_pseudodensity_4.g",
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

    # loading the pretrained model
    trainer.load_pretrained("../checkpoints/2021-03-22-14-02-20-test")

    calculator = AMPtorch(trainer)

    image = molecule('H2O')
    image.set_distance(0, 2, OH_bond_length, fix=0)
    image.set_angle(1, 0, 2, bond_angle)
    image.set_cell([10, 10, 10])
    image.center()
    image.set_calculator(calculator)

    return image.get_potential_energy()

def optimize_energy(x0):
    OH_bond_length = x0[0] 
    bond_angle = x0[1]
    images = read("../data/water.traj", index=":")
    images = [images[0]]

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
            "H": "../MCSH_potentials/H_pseudodensity_2.g",
            "O": "../MCSH_potentials/O_pseudodensity_4.g",
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

    # loading the pretrained model
    trainer.load_pretrained("../checkpoints/2021-03-22-14-02-20-test")

    calculator = AMPtorch(trainer)

    image = molecule('H2O')
    image.set_distance(0, 2, OH_bond_length, fix=0)
    image.set_angle(1, 0, 2, bond_angle)
    image.set_cell([10, 10, 10])
    image.center()
    image.set_calculator(calculator)

    print("Before minimizing forces: ")
    print(image.get_distance(0, 2))
    print(image.get_angle(1, 0, 2))

    print(image.get_potential_energy())
    
    dyn = BFGS(image)
    dyn.run(fmax=5e-2)

    print("Before minimizing forces: ")
    print(image.get_distance(0, 2))
    print(image.get_angle(1, 0, 2))

    return dyn

optimize_energy([1.2, 120])