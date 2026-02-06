# 
import os
import numpy as np

def reconstruct_full_configs(maps, nnk_log, interval, dims, scale, res_dir):

    # create dump file
    filepath = os.path.join(res_dir, "full_configs.dump")
    if os.path.exists(filepath):
        os.remove(filepath)
    dump = open(filepath, "a")

    jump_ids = nnk_log[:, 0].astype(np.int32)
    vacancy_id = jump_ids[0]
    num_of_atoms = len(maps)
    for i in range(len(jump_ids)):
        if i == 0:
            dump_config(maps, dump, i, num_of_atoms, dims, scale)
        else:
            jump_id = jump_ids[i]
            maps[vacancy_id][1], maps[jump_id][1] = maps[jump_id][1], maps[vacancy_id][1]

            if i % interval == 0:
                dump_config(maps, dump, i, num_of_atoms, dims, scale)


def reconstruct_effective_configs(maps, nnk_log, interval, dims, scale, res_dir):

    # create dump file
    filepath = os.path.join(res_dir, "effective_configs.dump")
    if os.path.exists(filepath):
        os.remove(filepath)
    dump = open(filepath, "a")

    jump_ids = nnk_log[:, 0].astype(np.int32)
    effective_ids = np.unique(jump_ids)
    effective_maps = {}
    for cur_id in effective_ids:
        effective_maps[cur_id] = maps[cur_id]

    vacancy_id = jump_ids[0]
    num_of_atoms = len(effective_maps)
    for i in range(len(jump_ids)):
        if i == 0:
            dump_config(effective_maps, dump, i, num_of_atoms, dims, scale)
        else:
            jump_id = jump_ids[i]
            effective_maps[vacancy_id][1], effective_maps[jump_id][1] = effective_maps[jump_id][1], effective_maps[vacancy_id][1]

            if i % interval == 0:
                dump_config(effective_maps, dump, i, num_of_atoms, dims, scale)

def reconstruct_vacancy_configs(maps, nnk_log, interval, dims, scale, res_dir):

    # create dump file
    filepath = os.path.join(res_dir, "vacancy_configs.dump")
    if os.path.exists(filepath):
        os.remove(filepath)
    dump = open(filepath, "a")

    jump_ids = nnk_log[:, 0].astype(np.int32)

    vacancy_id = jump_ids[0]
    num_of_atoms = 1
    
    effective_maps = {}
    for i, jump_id in enumerate(jump_ids):
        
        maps[vacancy_id][1], maps[jump_id][1] = maps[jump_id][1], maps[vacancy_id][1]
        effective_maps[vacancy_id] = maps[vacancy_id]
        dump_config(effective_maps, dump, i, num_of_atoms, dims, scale)

def reconstruct_unwrap_vacancy_configs(neuron_map, box_length, nnk_log, interval, dims, scale, res_dir):

    # create dump file
    filepath = os.path.join(res_dir, "vacancy_configs_unwrap.dump")
    if os.path.exists(filepath):
        os.remove(filepath)
    dump = open(filepath, "a")

    jump_ids = nnk_log[:, 0].astype(np.int32)

    '''
    vacancy_id = jump_ids[0]
    num_of_atoms = 1

    reference = maps[vacancy_id][1]
    #coords = reference.copy()
    #coords = dims / 2

    coords = np.array([0., 0., 0.])

    effective_maps = {}
    for i, jump_id in enumerate(jump_ids):
        
        maps[vacancy_id][1], maps[jump_id][1] = maps[jump_id][1], maps[vacancy_id][1]
        displacements = maps[vacancy_id][1] - reference
        displacements -= np.round(displacements / box_length) * box_length

        coords += displacements
    
        effective_maps[vacancy_id] = [maps[vacancy_id][0], coords]
        # print(displacements, coords, effective_maps)
        dump_config(effective_maps, dump, i, num_of_atoms, dims, scale)

        reference = maps[vacancy_id][1]
    '''

    vacancy_id = jump_ids[0]
    vacancy_disp = np.array([0, 0, 0,], dtype=np.float32)
    num_of_atoms = 1

    effective_maps = {}
    effective_maps[vacancy_id] = [neuron_map[vacancy_id][0], vacancy_disp]
    dump_config(effective_maps, dump, 0, num_of_atoms, dims, scale)    

    for i, jump_id in enumerate(jump_ids[1:]):

        cur_vac_disp = neuron_map[jump_id][1] - neuron_map[vacancy_id][1]
        cur_vac_disp -= np.round(cur_vac_disp / box_length) * box_length
        cur_vac_disp = cur_vac_disp.round(3)

        # update net vacancy/ion displacements
        vacancy_disp += cur_vac_disp

        effective_maps[vacancy_id] = [neuron_map[vacancy_id][0], vacancy_disp]
        dump_config(effective_maps, dump, i+1, num_of_atoms, dims, scale)

        # swap vacancy and ion
        neuron_map[vacancy_id][1], neuron_map[jump_id][1] = neuron_map[jump_id][1], neuron_map[vacancy_id][1]
 

def load_map(map_file):
    return np.load(map_file, allow_pickle=True).item()

def load_nnk_log(nnk_log_file):
    return np.loadtxt(nnk_log_file, delimiter=" ")

def dump_config(maps, print_f, frame, num_of_atoms, dims, scale):

    # print prefix info
    print("ITEM: TIMESTEP", file=print_f)
    print("{:.0f}".format(frame), file=print_f)
    print("ITEM: NUMBER OF ATOMS", file=print_f)
    print("{:.0f}".format(num_of_atoms), file=print_f)
    print("ITEM: BOX BOUNDS pp pp pp", file=print_f)
    for i in range(3):
        print("0 {:.2f}".format(dims[i]), file=print_f)
    print("ITEM: ATOMS id type x y z", file=print_f) 

    # print atomic info
    for map_id in np.sort([*maps.keys()]):
        map_val = maps[map_id]
        atom_id, atom_type, index = map_id, map_val[0], map_val[1]
        x, y, z = index[0] * scale, index[1] * scale, index[2] * scale
        print("{:.0f} {:.0f} {:.2f} {:.2f} {:.2f}".format(atom_id, atom_type, x, y, z), file=print_f)
