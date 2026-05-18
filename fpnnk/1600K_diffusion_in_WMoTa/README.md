# First-principle Neural Network Kinetics (FPNNK) Simulation of Vacancy Diffusion in W-Mo-Ta Alloys at 1600 K.

## Overview

The folder `1600K_diffusion_in_WMoTa` demonstrates an example of using the First-principle Neural Network Kinetics (FPNNK) computational scheme to simulate vacancy diffusion in a ternary refractory alloy (W-Mo-Ta) at 1600 K. The basic settings are listed in the following table.


| Property             | Value             |
| -------------------- | ----------------- |
| Alloy system         | Equimolar W-Mo-Ta |
| Atomistic model size | 16,000 atoms      |
| Number of vacancies  | 1                 |
| Temperature          | 1600 K            |
| Simulation length    | 1,000 steps       |


## File Structure

```
.
├── user_inp                     # All user-facing simulation parameters
├── MoTaW.dump                   # Dump file of W-Mo-Ta alloy
├── nnk_simu.py                  # Main simulation driver
├── postprocess.py               # Construct vacancy config dump file from simulation log file
├── plot_vacancy_trajectory.py   # Visualize vacancy diffusion trajectory in 3d space
└── run.sh                       # Run simulation, postprocessing and plotting in one shot
```

---

## Running the Simulation

### Step 1 — Configure `user_inp`

All simulation parameters are controlled through the `user_inp` file. Edit settings as needed to perform simulations at different conditions. Check the [INPUT Parameter Reference](#input-parameter-reference) for more details about the meaning of each parameter in `user_inp`.

### Step 2 — Run the simulation

```bash
python nnk_simu.py user_inp
```

On completion, the simulation log file is saved to `res_dir/nnk.log`. The `nnk.log` consists of two columns, indicating jumping atom id and time.

### Step 3 — Construct vacancy dump file

```bash
python postprocess.py
```

Reads `res_dir/nnk.log` and produces `res_dir/vacancy_configs_unwrap.dump`. The dump file consists of vacancy positions at different steps (vacancy positions are already unwrapped).

### Step 4 — Visualize the vacancy trajectory

```bash
python plot_vacancy_trajectory.py
```

Reads `res_dir/vacancy_configs_unwrap.dump` and produces `vacancy_trajectory.png`: a 3D scatter/line plot of the vacancy path colored sequentially by timestep. Note: the vacancy diffusion trajectories can also be directly visualized using OVITO.

### Alternatively, all four steps can be run in one shot using:

```bash
bash run.sh
```

---

## Expected Output


| File                                  | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| `res_dir/nnk.log`                     | Per-step log recording vacancy atom ID and KMC time at each step   |
| `res_dir/vacancy_configs_unwrap.dump` | LAMMPS-format trajectory of the vacancy position at every KMC step |
| `vacancy_trajectory.png`              | 3D visualization of the complete vacancy diffusion trajectory      |


---

## Expected Run Time

Running 1,000 KMC steps on a standard desktop computer (GPU required for inference) takes approximately **15 seconds**.

---

## INPUT Parameter Reference

### Initial Configuration


| Parameter          | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `init_config_dump` | Path to the initial atomic configuration in LAMMPS dump format |
| `dim_row`          | Starting line of simulation box description in dump file       |
| `dim_row_num`      | Number of lines for simulation box description in dump file    |
| `config_row`       | Starting line of atoms information in dump file                |
| `num_of_atoms`     | Total number of atoms in the simulation supercell              |


### Neuron Map Construction


| Parameter    | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `cutoff`     | Cutoff distance (Å) for encoding local environments                |
| `voxel_size` | Spatial voxel resolution for environment encoding (Å)              |
| `flatten`    | Flatten neuron map to 1D before ML inference (`1` = yes, `0` = no) |


### KMC Simulation


| Parameter      | Description                           |
| -------------- | ------------------------------------- |
| `init_step`    | Starting KMC step index               |
| `num_of_steps` | Total number of KMC steps to simulate |
| `temperature`  | Simulation temperature (K)            |


### Vacancy Initialization


| Parameter         | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `random_vacancy`  | Select the initial vacancy randomly (`1`) or by explicit ID (`0`)     |
| `vacancy_id`      | Atom ID to remove as the vacancy; used only when `random_vacancy = 0` |
| `dump_vacancy_id` | Write vacancy atom ID at the first line of nnk.log (`1` = yes)        |


### Neural Network Model Weights


| Parameter         | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `ml_model_weight` | Path to pre-trained neural network model weights (`.pt` file) for barrier prediction |


### Output and Parallelism


| Parameter     | Description                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `log_file`    | Filename for the per-step simulation log including jumping atom id and time |
| `res_dir`     | Directory where all output files are written                                |
| `num_of_cpus` | Number of CPU cores to use                                                  |


