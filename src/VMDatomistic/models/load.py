import torch
import os


def load_multifidelity_m3gnet(device: torch.device('cuda'), dtype: torch.float32, compute_forces: bool = False,
                              compute_stress: bool = False) -> torch.jit.ScriptModule:
    """
    :param device: torch.device
    :param dtype: dtype, default torch.float32
    :param compute_forces: Whether the model returns forces
    :param compute_stress: Wether the model returns stress
    :return: TorchScript module


    The forward method of the model has the following input signature:

    def forward(
        coordinates: Tensor,  # [N_atoms, 3] in Angstrom
        species: Tensor,  # [N_atoms,]
        cell: Tensor,  # [N_batch, 3, 3] in Angstrom
        edge_index: Tensor,  # [2, N_edges]
        shifts: Tensor,  # [N_edges, 3] in Angstrom
        batch: Tensor,  # [N_atoms,]
        dataset_index: Tensor  # [N_atoms,]
    ) -> Dict[str, Tensor]

    The model returns a dictionary with keys 'energy' [eV], 'node_energies' [eV], 'forces' (optional) [eV/Ang],
    'stress' (optional) [eV/Ang^3]
    """
    path = os.path.join(os.path.dirname(__file__), "./checkpoints/M3GNet_PBE_r2SCAN_epoch_100.pt")
    model = torch.jit.load(path, map_location=device).eval()
    model.to(device=device, dtype=dtype)
    model.compute_forces = compute_forces  # disable/enable computation of forces
    model.compute_stress = compute_stress  # disable/enable computation of stress
    return model
