## First-principle Neural Network Kinetics (FPNNK)

This repository contains the computational framework, First-principle Neural Network Kinetics, for implementing vacancy diffusion simulations with DFT-level predictive accuracy. The FPNNK scheme can efficiently simulate vacancy diffusion through combining deep neural network, which is trained on the diffusion barrier dataset from density functional theory calculations, and kinetic Monte Carlo. The deep neural network predicts the path-dependent energy barriers from local atomic environment encoded by on-lattice representation. The kinetic Monte Carlo samples the diffusion jump direction and timescale based on neural network predicted energy barriers.
---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)
- [Reproducing Results](#reproducing-results)
- [License](#license)

---

## System Requirements

### Software Dependencies

| Package | Version Tested | Notes |
|---|---|---|
| Python | 3.12.4 | Required |
| PyTorch | 2.4.0 | Deep neural network backend |
| CUDA | 12.4.0 | GPU acceleration |
| NumPy | 1.26.4 | Array operations |
| pandas | 2.2.2 | Data handling |
| matplotlib | 3.10.9 | Visualization |
| scipy | 1.13.1 | Scientific computing |
| torchsummary | 1.5.1 | Model summary utility |
| VASP | 5.x / 6.x | Required for DFT/NEB calculations only |

> Base environment: PyTorch 2.4.0, CUDA 12.4.0, Mambaforge 24.5.0-0, Python 3.12.4, Ubuntu 22.04

### Operating Systems Tested

- Linux (Ubuntu 22.04)

### Hardware Requirements

- Standard desktop or laptop computer
- GPU required for neural network training (CUDA 12.4.0-compatible)
- Typical RAM: ≥ 8 GB

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UCICaoLab/First-principle-NNK.git
cd First-principle-NNK/fpnnk
```

### 2. Install the package

**Standard installation:**

```bash
pip install .
```

**Development installation** (recommended if you plan to modify source code):

```bash
pip install -e .
```

**Typical installation time:** ~2–5 minutes on a normal desktop computer.

---

## Repository Structure

```
First-principle-NNK/
├── DFT_NEB_input/        # VASP input files for NEB diffusion barrier calculations
├── DFT_training_data/    # Atomistic structures and DFT-computed diffusion barriers
└── fpnnk/
    ├── src/              # Source code for the FPNNK framework
    ├── model_weights/    # Pre-trained neural network model weights
    └── example/
        ├── MoTaW.dump        # Example atomic model (equimolar Mo-Ta-W system)
        ├── user_inp          # Example input parameter file
        ├── nnk_simu.py       # Main simulation script
        └── postprocess/
            └── postprocess.py  # Post-processing script to recover atomic configurations
```

---

## Demo

A complete demo is provided in the `fpnnk/example/` directory using an equimolar **Mo-Ta-W** alloy system.

### Run the demo

```bash
cd fpnnk/example
python nnk_simu.py user_inp
```

where `user_inp` is the input parameter file specifying simulation settings (e.g., temperature, number of steps, atomic model path).

**Expected output:** An `nnk.log` file will be generated in the `res_data/` directory. Each row contains:
- Column 1: ID of the jumping atom
- Column 2: Jump time (physical time in seconds)

**Expected run time:** ~a few minutes on a normal desktop computer for a short demo simulation.

### Post-processing

The `nnk.log` file stores only atom IDs and times to save storage. To recover full atomic configurations:

```bash
cd fpnnk/example/postprocess
python postprocess.py
```

This script rebuilds atomic configurations (positions of all atoms at each step) from the log file.

---

## Instructions for Use

### Running vacancy diffusion simulations on your own data

1. **Prepare your atomic model** — provide a LAMMPS `.dump` file with your alloy system.
2. **Edit `user_inp`** — specify your atomic model file path, temperature, number of kMC steps, and other parameters.
3. **Run the simulation:**

```bash
python nnk_simu.py user_inp
```

4. **Post-process results** using the provided `postprocess.py` script to extract atomic trajectories and calculate diffusion coefficients or other properties.

### Computing new DFT training data (optional)

If you wish to retrain the neural network for a new alloy system:

1. Use the VASP input files in `DFT_NEB_input/` as templates to compute NEB diffusion barriers for your system.
2. Add computed barriers and structures to `DFT_training_data/`.
3. Retrain the neural network using the training scripts in `fpnnk/src/`.

---

## Reproducing Results

To reproduce the quantitative results reported in the manuscript:

1. Install the package following the [Installation](#installation) instructions above.
2. Navigate to `fpnnk/example/` and run the demo simulation as described in the [Demo](#demo) section.
3. Use `postprocess.py` to extract atomic configurations and compute mean-squared displacement (MSD) and diffusion coefficients.

Detailed pseudocode and descriptions of the FPNNK algorithm are provided in the **Methods** section and **Extended Data** of the associated manuscript.

---

## License

Please see the `LICENSE` file in this repository for terms of use.

---

## Citation

If you use this code, please cite the associated manuscript (citation details to be added upon publication).

---

## Contact

For questions or issues, please open a GitHub Issue or contact the corresponding author via the associated manuscript.
