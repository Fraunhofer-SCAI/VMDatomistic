from __future__ import annotations

from typing import Union, Optional
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..data.transform import PeriodicRadiusGraph
from ..models.load import load_multifidelity_m3gnet


class PeriodicVMDCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self,
                 model: Union[torch.nn.Module, torch.jit.ScriptModule],
                 dataset_index: int = 0,
                 compute_forces: bool = True,
                 compute_stress: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):
        super().__init__(**kwargs)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_index = dataset_index
        self.dtype = dtype
        self.device = device
        self.model = model.to(device=device, dtype=dtype)
        self.cutoff = model.cutoff
        if compute_forces:
            self.model.compute_forces = True
            self.implemented_properties.append("forces")
        if compute_stress:
            self.model.compute_stress = True
            self.implemented_properties.append("stress")
        self.compute_forces = compute_forces
        self._jit_optimize = False
        self.graph_computer = PeriodicRadiusGraph(cutoff=model.cutoff)

    @classmethod
    def multifidelity_m3gnet(cls,
                             fidelity: str = "r2SCAN",
                             compute_forces: bool = True,
                             compute_stress: bool = True,
                             device: Optional[torch.device] = None,
                             dtype: torch.dtype = torch.float32,
                             **kwargs) -> PeriodicVMDCalculator:

        assert fidelity.lower() in ["pbe",
                                    "r2scan"], f"Fidelity must be either 'PBE' or 'r2SCAN'. " \
                                               f"{fidelity} is not supported."
        fidelity_mapping = {"pbe": 0, "r2scan": 1}
        dataset_index = fidelity_mapping[fidelity.lower()]
        model = load_multifidelity_m3gnet(device=device,
                                          dtype=dtype,
                                          compute_forces=compute_forces,
                                          compute_stress=compute_stress)
        print(f"Created calculator instance that evaluates the M3GNet model using '{fidelity}' fidelity settings.")
        calc = cls(model=model,
                   dataset_index=dataset_index,
                   compute_forces=compute_forces,
                   compute_stress=compute_stress,
                   device=device,
                   dtype=dtype,
                   **kwargs)
        return calc

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):

        super().calculate(atoms, properties, system_changes)

        sample = {
            "pos": torch.from_numpy(atoms.positions).to(dtype=self.dtype, device=self.device),
            "z": torch.from_numpy(atoms.numbers).to(dtype=torch.long, device=self.device).view(-1),
            "cell": torch.from_numpy(atoms.cell[:]).to(dtype=self.dtype, device=self.device).unsqueeze(0),
            "pbc": torch.from_numpy(atoms.pbc).to(dtype=torch.bool, device=self.device),
        }

        sample = self.graph_computer(sample)

        sample.update({"batch": torch.zeros_like(sample["z"]),
                       "dataset_index": torch.full_like(sample["z"], fill_value=self.dataset_index)})

        with torch.jit.optimized_execution(self._jit_optimize):
            prediction = self.model.forward(
                coordinates=sample["pos"],  # [N_atoms, 3] #
                species=sample["z"],  # [N_atoms,]
                cell=sample["cell"],  # [N_batch, 3, 3]
                edge_index=sample["edge_index"],  # [2, N_edges]
                shifts=sample["shifts"],  # [N_edges, 3]
                batch=torch.zeros_like(sample["z"]),  # [N_atoms,]
                dataset_index=torch.full_like(sample["z"], fill_value=self.dataset_index)  # [N_atoms,]
            )

        # Extract energy, forces, and stress from the model's output
        energy = prediction.get("energy")
        forces = prediction.get("forces")
        stress = prediction.get("stress")

        self.results["energy"] = energy.item()

        # Check if the outputs are available
        if energy is None:
            raise ValueError("The model did not return 'energy' in its output.")
        if "forces" in properties:
            if forces is None:
                raise ValueError("The model did not return 'forces' in its output.")
            else:
                self.results["forces"] = forces.detach().cpu().numpy()
        if "stress" in properties:
            if stress is None:
                raise ValueError("The model did not return 'stress' in its output.")
            else:
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress.view(3, 3).detach().cpu().numpy())
