# postprocess output from nnk simulation
import numpy as np
import shutil

import nnk.process_module 

dump_file_name, nnk_log_file_name = "../MoTaW.dump", "../res_data/nnk.log"
dump_data = np.loadtxt(dump_file_name, delimiter=" ", skiprows=9)

# log_data = np.loadtxt(nnk_log_file_name, delimiter=" ",)

neuron_map = {}
for i in range(len(dump_data)):
    neuron_map[int(dump_data[i, 0])] = [int(dump_data[i, 1]), dump_data[i, 2:].round(2)]

nnk_log = nnk.process_module.load_nnk_log(nnk_log_file_name)

# simulation box dimensions
dims = [64.80 for _ in range(3)]     # simulation box dimensions
dims = np.array(dims)

# dump interval 
interval = 1

# scale or voxel size
scale = 1.0 #1.62

# case 1: print vacancy 
# nnk.process_module.reconstruct_vacancy_configs(neuron_map, nnk_log, interval, dims, scale, "./")

# case 2: print vacancy without pbc conditions
# nnk.process_module.reconstruct_unwrap_vacancy_configs(neuron_map, dims, nnk_log, interval, dims, scale, "./")

# case 3: print atoms that moved during simulations
# nnk.process_module.reconstruct_effective_configs(neuron_map, nnk_log, interval, dims, scale, "./")

# case 4: print all atoms
nnk.process_module.reconstruct_full_configs(neuron_map, nnk_log, interval, dims, scale, ".")

#shutil.move("vacancy_configs.dump", "configs/vacancy_configs.dump.{}".format(index+1))
#shutil.move("vacancy_configs_unwrap.dump", "configs/vacancy_configs_unwrap.dump.{}".format(index))


