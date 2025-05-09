from ase.io import read, Trajectory
from ase.optimize import BFGS
import torch

from VMDatomistic.calculators import PeriodicVMDCalculator

atoms = read('data/example_structures.extxyz', index=10)

calculator = PeriodicVMDCalculator.multifidelity_m3gnet(fidelity="r2SCAN", dtype=torch.float64)
atoms.calc = calculator

traj = Trajectory('optimized_structure.traj', 'w', atoms)

optimizer = BFGS(atoms, trajectory=traj)

optimizer.run(fmax=0.01)
