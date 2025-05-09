# Multi-fidelity M3GNet for material simulations

The goal of this repository is to showcase the `trainable data embedding` approach presented in our paper [1] for
training atomistic machine learning models on multiple reference methods simultaneously.
This repository contains a TorchScript model for evaluating an implementation of the M3GNet [2] model that was trained
on two different fidelities found in the MatPES dataset [3] (PBE and r2SCAN). In addition, we provide an ASE calculator.

A scalable multi-GPU version of this model and other machine learning force fields will be available in our massively
parallel
software
package [Tremolo-X](https://www.scai.fraunhofer.de/en/business-research-areas/virtual-material-design/products/tremolo-x.html), soon. News will be announced here and on the website. Currently, the website is expecting major changes and will be
updated in the next weeks.

## Installation:

The model can be installed with pip:

```python
pip install git+https://github.com/Fraunhofer-SCAI/VMDatomistic
```

Dependencies are `torch>=2.1.0`, `ase`, `pymatgen` and optionally `torch_geometric` for batched inference. You can
install `torch_geometric` automatically with:

```python
pip install git+https://github.com/Fraunhofer-SCAI/VMDatomistic[pyg]
```

# Examples:

The folder `examples` contains example scripts, including single sample inference and batched inference.
For batched inference, we rely on the mini-batching concept from PyTorch Geometric. Although batched inference is possible without installing `torch_geometric`, its installation is recommended for efficient handling of graph data.

## ASE calculator

The fidelity for the calculator can be changed between `r2SCAN` (default) and `PBE` when loading the calculator instance.

```python
from ase.io import read
import torch
from VMDatomistic.calculators import PeriodicVMDCalculator

dtype = torch.float64
device = torch.device('cuda')
atoms = read('path/to/structure')
calculator = PeriodicVMDCalculator.multifidelity_m3gnet(fidelity="r2SCAN", dtype=dtype, device=device)
atoms.calc = calculator
# perform MD simulation or geometry optimization in the following:
...
```

# Citation

When using the model for your work, please cite our work with:

```
@misc{OSH2025, 
title={Trainable Data Embeddings Enable Multi-Fidelity Learning}, 
DOI={10.26434/chemrxiv-2025-vx7nx}, 
url={https://doi.org/10.26434/chemrxiv-2025-vx7nx}, 
author={Oerder, Rick and Schmieden, Gerrit and Hamaekers, Jan}, 
year={2025},
note={ChemRxiv Preprint.}} 
```

Please cite [2] and [3], as well. 

# References

[1] Oerder, R., Schmieden, G., & Hamaekers, J. (2025). Trainable Data Embeddings Enable Multi-Fidelity Learning.
ChemRxiv Preprint. DOI: [10.26434/chemrxiv-2025-vx7nx](https://doi.org/10.26434/chemrxiv-2025-vx7nx).

[2] Chen, C., & Ong, S.P. (2023). A universal graph deep learning interatomic potential for the periodic table. Nature
Computational Science, 2, 718–728. DOI: [10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3).

[3] Kaplan, A. D., Liu, R., Qi, J., Ko, T. W., Deng, B., Riebesell, J., Ceder, G., Persson, K. A., & Ong, S. P. (2025).
A Foundational Potential Energy Surface Dataset for Materials. arXiv:2503.04070.
DOI: [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070).


# License
 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">The content of this repository is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>