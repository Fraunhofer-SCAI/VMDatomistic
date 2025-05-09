import torch
from ase.io import read
from .transform import PeriodicRadiusGraph


def load_data(path: str, cutoff: float, dtype: torch.dtype):
    data_list = []
    grapher_computer = PeriodicRadiusGraph(cutoff)
    atoms_list = read(path, index=":")

    for i, atoms in enumerate(atoms_list):
        atom_data = {
            "pos": torch.from_numpy(atoms.positions).to(dtype=dtype),
            "z": torch.from_numpy(atoms.numbers).to(dtype=torch.long).view(-1),
            "cell": torch.from_numpy(atoms.cell[:]).to(dtype=dtype).unsqueeze(0),
            "pbc": torch.from_numpy(atoms.pbc).to(dtype=torch.bool)
        }
        atom_data = grapher_computer(atom_data)
        data_list.append(atom_data)
    return data_list
