## First-principle Neural Network Kinetics (FPNNK) | [https://github.com/UCICaoLab/First-principle-NNK](https://github.com/UCICaoLab/First-principle-NNK)

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


| Package      | Version Tested | Notes                                  |
| ------------ | -------------- | -------------------------------------- |
| Python       | 3.12.4         | Required                               |
| PyTorch      | 2.4.0          | Deep neural network backend            |
| CUDA         | 12.4.0         | GPU acceleration                       |
| NumPy        | 1.26.4         | Array operations                       |
| pandas       | 2.2.2          | Data handling                          |
| matplotlib   | 3.10.9         | Visualization                          |
| scipy        | 1.13.1         | Scientific computing                   |
| torchsummary | 1.5.1          | Model summary utility                  |
| VASP         | 5.x / 6.x      | Required for DFT/NEB calculations only |


### Operating Systems Tested

- Linux

### Hardware Requirements

- Standard desktop or laptop computer
- GPU required

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

The installation on a normal desktop computer can be completed in one second.

---

## Repository Structure

```
First-principle-NNK/
├── DFT_NEB_input/              # VASP input files for NEB diffusion barrier calculations
├── DFT_training_data/          # Atomistic structures and DFT-computed diffusion barriers
└── fpnnk/
    ├── src/                    # Source code for the FPNNK framework
    ├── model_weights/          # Pre-trained neural network model weights
    └── 1600K_diffusion_in_WMoTa/   # Example: vacancy diffusion in Mo-Ta-W at 1600K
        ├── MoTaW.dump              # Atomic model (equimolar Mo-Ta-W system)
        ├── user_inp                # Input parameter file
        ├── nnk_simu.py             # Main simulation script
        ├── postprocess.py          # Post-processing script
        ├── plot_vacancy_trajectory.py    # Plot vacancy diffusion trajectory
        ├── vacancy_trajectory.png        # Vacancy diffusion trajectory
        ├── run.sh                  # Bash file for running nnk simulation, postprocessing and plotting sequentially
        └── res_dir/                # Output directory
```

---

## Demo

A complete demo is provided in the `fpnnk/1600K_diffusion_in_WMoTa/` directory using an equimolar **Mo-Ta-W** alloy system. Check `fpnnk/1600K_diffusion_in_WMoTa/README.md` for detailed instructions.

---

## Instructions for Use

### Running vacancy diffusion simulations on your own data

1. **Prepare your atomic model** — provide a LAMMPS `.dump` file with your alloy system.
2. **Edit `user_inp`** — specify your atomic model file path, temperature, number of kMC steps, and other parameters. Check `fpnnk/1600K_diffusion_in_WMoTa/README.md` for more information about various parameters in simulation settings.
3. **Run the simulation:**

```bash
python nnk_simu.py user_inp
```

4. **Post-process results** using the provided `postprocess.py` script to extract atomic trajectories and compute related properties.

---

## Reproducing Results

To reproduce the quantitative results reported in the manuscript:

1. Install the package following the [Installation](#installation) instructions above.
2. Navigate to `fpnnk/1600K_diffusion_in_WMoTa/` and run the demo simulation as described in the [Demo](#demo) section.
3. Adjust simulation input parameters in `user_inp` to reproduce results in the manuscript where the corresponding simulation settings are provided.

Detailed descriptions of the FPNNK algorithm are provided in the **Methods** section of the associated manuscript.

---

## License

Please see the `LICENSE` file in this repository for terms of use.

---

## Citation

If you use this code, please cite the associated manuscript (citation details to be added upon publication).

---

## Contact

For questions or issues, please open a GitHub Issue or contact the corresponding author via the associated manuscript.