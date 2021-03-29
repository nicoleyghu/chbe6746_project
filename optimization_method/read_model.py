import numpy as np
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from ase.build import molecule

def load_trainer(checkpoint_path):
    trainer = AtomsTrainer()
    # loading the pretrained model
    trainer.load_pretrained(checkpoint_path)
    return trainer

def calculate_energy(x0,trainer):
    OH_bond_length = x0[0] 
    bond_angle = x0[1]

    calculator = AMPtorch(trainer)

    image = molecule('H2O')
    image.set_distance(0, 2, OH_bond_length, fix=0)
    image.set_angle(1, 0, 2, bond_angle)
    image.set_cell([10, 10, 10])
    image.center()
    image.set_calculator(calculator)

    return image.get_potential_energy()

# def optimize_energy(x0):
#     OH_bond_length = x0[0] 
#     bond_angle = x0[1]
#     trainer = AtomsTrainer()

#     # loading the pretrained model
#     trainer.load_pretrained("../checkpoints/2021-03-22-14-02-20-test")

#     calculator = AMPtorch(trainer)

#     image = molecule('H2O')
#     image.set_distance(0, 2, OH_bond_length, fix=0)
#     image.set_angle(1, 0, 2, bond_angle)
#     image.set_cell([10, 10, 10])
#     # fix the OH. 
#     fix_OH = FixAtoms([0, 1])
#     image.set_constraint(fix_OH)
#     image.center()
#     image.set_calculator(calculator)

#     print("Before minimizing forces: ")
#     print(image.get_distance(0, 2))
#     print(image.get_angle(1, 0, 2))

#     print(image.get_potential_energy())
    
#     dyn = BFGS(image)
#     dyn.run(fmax=1e-3)

#     print("Before minimizing forces: ")
#     print(image.get_distance(0, 2))
#     print(image.get_angle(1, 0, 2))

#     return dyn

# optimize_energy([1.2, 120])