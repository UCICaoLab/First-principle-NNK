rm -rf res_dir

python nnk_simu.py user_inp

python postprocess.py

python plot_vacancy_trajectory.py
