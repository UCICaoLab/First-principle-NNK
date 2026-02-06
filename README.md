## First-principle Neural Network Kinetics (FPNNK)
This repository contains the computational framework, First-principle Neural Network Kinetics, for implementing vacancy diffusion simulations with DFT-level predictive accuracy. The FPNNK scheme can efficiently simulate vacancy diffusion through combining deep neural network, which is trained on the diffusion barrier dataset from density functional theory calculations, and kinetic Monte Carlo. The deep neural network predicts the path-dependent energy barriers from local atomic environment encoded by on-lattice representation. The kinetic Monte Carlo samples the diffusion jump direction and timescale based on neural network predicted energy barriers.

## Repository Structure
- `DFT_NEB_input/` - Input files for computing diffusion barriers using the NEB method in VASP.
- `DFT_training_data/` - Source data including atomistic structures and diffusion barriers. 
- `fpnnk/` - Source code of the FPNNK computational framework.

## Installation
The package can be installed following the following steps.

#### Clone the repository
```bash
git clone git@github.com:UCICaoLab/First-principle-NNK.git
cd First-principle-NNK/fpnnk
```

#### Install the package
For standard installation,
```bash
pip install .
```
For development installation (if you need to edit the source code frequently), 
```bash
pip install -e .
```

## Usage
In `fpnnk/` directory, three subdirectories exists:
- `src/` - Stores the source code.
- `model_weights/` - Stores pre-trained deep neural network model weights.
- `example/` - Provides examples for runing some quick tests.

The `example/` directory provides scripts for performing diffusion simulations in a equimolar Mo-Ta-W atomic system (`MoTaW.dump` is the atomic model dump file). 

#### Run vacancy diffusion simulations
To run the simulation, typing the following commands:
```bash
python nnk_simu.py user_inp
```
where `user_inp` includes all input parameters needed for the simulation.

#### Generate outputs
The program will generate an output file `nnk.log` in the `res_data/` directory. In `nnk.log`, the first column denotes the id of jumping atoms and the second column denotes the jump time, which could be used for extracting spatial and temporal information for further analysis.

#### Postprocess
The `nnk.log` only saves jumping atom id and time in order to save the time and space for printing atomic configuration information at each step. Thus, additional postprocessing is needed to extract and recover atomic configurations from `nnk.log`. In the `postprocess/` inside the `example`, a sample script is provided to extract rebuild atomic configurations from generated log file.

Type the following command to generate vacancy configurations or atomic configurations from simulations:
```bash
python postprocess.py
```
