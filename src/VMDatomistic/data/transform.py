import numpy as np
import torch
from pymatgen.optimization.neighbors import find_points_in_spheres


class PeriodicRadiusGraph:
    """
    Compute neighborlist with PyMatgen
    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff: Cutoff radius for neighbor search.
        """

        super().__init__()
        self.cutoff = cutoff

    def __call__(self, data):
        coordinates = data["pos"]
        cell = data["cell"]
        np_pbc = np.array([1, 1, 1], dtype=int)

        device = coordinates.device
        dtype = coordinates.dtype

        np_cell = cell.view(3, 3).detach().cpu().numpy().astype(np.float64)
        np_coordinates = coordinates.detach().cpu().numpy().astype(np.float64)

        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            np_coordinates,
            np_coordinates,
            r=self.cutoff,
            pbc=np_pbc,
            lattice=np_cell,
        )

        exclude_self = (src_id != dst_id) | (bond_dist > 1e-8)
        src_id, dst_id, shifts = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
        )

        shifts = shifts @ np_cell
        shifts = torch.from_numpy(shifts).to(device=device, dtype=dtype)
        edge_index = np.vstack([src_id, dst_id])
        edge_index = torch.from_numpy(edge_index).long().to(device)

        update_dict = {
            "edge_index": edge_index,
            "shifts": shifts,
        }
        data.update(update_dict)

        return data