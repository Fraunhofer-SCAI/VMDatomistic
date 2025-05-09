import torch
from tqdm import tqdm
from VMDatomistic import load_data, load_multifidelity_m3gnet

# specify dtype and device:
dtype = torch.float32
device = torch.device("cuda")

# load model with torchscript:
dataset_index = 1  # mapping from dataset_idx to DFT reference: '0' : 'PBE', '1': 'r2SCAN'
model = load_multifidelity_m3gnet(dtype=dtype, device=device, compute_forces=False,
                                  compute_stress=False)

# load data
data_path = "data/example_structures.extxyz"
data_list = load_data(path=data_path, cutoff=model.cutoff, dtype=dtype)

# single sample inference:
for sample in tqdm(data_list):

    # move data to device
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sample[key] = val.to(device)

    # prediction:
    prediction = model.forward(
        coordinates=sample["pos"],  # [N_atoms, 3]
        species=sample["z"],  # [N_atoms,]
        cell=sample["cell"],  # [N_batch, 3, 3]
        edge_index=sample["edge_index"],  # [2, N_edges]
        shifts=sample["shifts"],  # [N_edges, 3]
        batch=torch.zeros_like(sample["z"]),  # [N_atoms,]
        dataset_index=torch.full_like(sample["z"], fill_value=dataset_index)  # [N_atoms,]
    )