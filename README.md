## First-principle Neural Network Kinetics (FPNNK)
This repository contains the computational framework, First-principle Neural Network Kinetics, for implementing vacancy diffusion simulations with DFT-level predictive accuracy. The FPNNK scheme can efficiently simulate vacancy diffusion through combining deep neural network, which is trained on the diffusion barrier dataset from density functional theory calculations, and kinetic Monte Carlo. The deep neural network predicts the path-dependent energy barriers from local atomic environment encoded by on-lattice representation. The kinetic Monte Carlo samples the diffusion jump direction and timescale based on neural network predicted energy barriers.

## Repository Structure
- `DFT_NEB_input/` - Input files for computing diffusion barriers using the NEB method in VASP.
- `DFT_training_data/` - Source data including atomistic structures and diffusion barriers. 
- `fpnnk/` - Source code of the FPNNK computational framework.

## Installation
The package can be installed following the following steps.

1. Download the python package.
2. cd First-principle-NNK/fpnnk
3. pip install . or pip install -e . if you would like to frequently edit soruce code

## Usage
In fpnnk directory, three subdirectories exists:
1. src: storing the source code;
2. model_weights: storing pre-trained deep neural network model weights;
3. example: providing examples for runing some quick testing.

The example folder provides scripts for performing diffusion simulations in a equimolar Mo-Ta-W atomic system (MoTaW.dump is the atomic model dump file). 

To run the simulation, typing the following commands:
python nnk_simu.py user_inp
where "user_inp" includes all input parameters for the simulation.

The program will generate the output file "nnk.log" in the folder called res_data. In nnk.log, the first column denotes the id of jumping atom and the second column denotes the jump time, which could be used for extracting spatial and temporal information for further analysis.

The python script postprocess.py is used for extracting useful information from simulation outputs such as reconstructing and dumping atomic configurations. 
