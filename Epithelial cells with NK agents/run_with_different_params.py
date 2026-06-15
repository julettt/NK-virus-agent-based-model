import numpy as np
import os
from nk_epith_model import run_with_params

BASE_params = dict(time_max = 120.0, virus_delay = 0.0, NK_delay = 4.0, max_steps = 1_000_000, ani_step_save = 10_000,
                   grid_size = 100, max_NK = 3, a = 1.0, b = 3.0, max_inf_state = 30,
                   t_prog = 1.0, beta_spread = 0.7502, t_spread = 119.6937, M_I_spread = 3, M_I_death = 3,
                   gamma_ind = 0.0514, t_ind = 399.1724, gamma_dep = 0.0520, t_dep = 21.3922, 
                   MOI = 0.1, NK_ratio = 1.0, 
                   with_death_dep = True, with_death_ind = True, with_infection_spread = True, with_infection_progression = True)


#parameter to change:
PARAM_NAME = "NK_delay"
PARAM_VALUES = [4.0]

#virus or NK impact - REMEMBER TO CHANGE TO (True, False) AFTER MAKING A CHANGE
NK_impact = True
virus_impact = False

os.makedirs("results", exist_ok = True)
print(f"Simluations to make: {len(PARAM_VALUES)}")

for i, v in enumerate(PARAM_VALUES):
    params = BASE_params.copy()
    params[PARAM_NAME] = v

    if NK_impact:
        if params["NK_delay"] > params["time_max"]: #no nk
            out_path = f"plot results/without NK, {PARAM_NAME}={v}.npz"
        else:
            out_path = f"plot results/with NK, {PARAM_NAME}={v}.npz"

    if virus_impact:
        if params["virus_delay"] > params["time_max"]: #no virus
            out_path = f"plot results/without virus, {PARAM_NAME}={v}.npz"
        else:
            out_path = f"plot results/with v, {PARAM_NAME}={v}.npz"


    print(f"[{i+1}/{len(PARAM_VALUES)}]; starting: {PARAM_NAME}={v}")
    result = run_with_params(params)
    print(f"done, t = {result["total_time"]}, steps = {result["total_steps"]}")

    np.savez_compressed(out_path, 
                        stats_array = result["stats_array"],
                        frames_NK = result["frames_NK"],
                        frames_epith = result["frames_epith"],
                        **{k: np.array(v) for k, v in params.items()})
    
print("All done.")
