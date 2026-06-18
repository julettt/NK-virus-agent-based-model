import numpy as np
import os
from nk_epith_model_v2 import run_with_params


def run_for_params(idx, t_spread_val = 50, c_I_NK_val = 0, NK_ratio_val = 0.1, NK_delay_val = 4.0):

    BASE_params = dict(time_max = 10_000.0, virus_delay = 0.0, NK_delay = NK_delay_val, max_steps = 50_000_000, ani_step_save = 10_000,
                    grid_r = 50, grid_c = 200, max_NK = 3, a = 1.0, b = 3.0, max_inf_state = 15,
                    t_prog = 1, beta_spread = 1, t_spread = t_spread_val, M_I = 5, M_D = 5,
                    c_0 = 0, c_I = 0.03, c_NK = 0, c_I_NK = c_I_NK_val, 
                    MOI = 1, NK_ratio = NK_ratio_val, 
                    with_death_I = True, with_death_NK = True, with_infection_spread = True, with_infection_progression = True)

    #parameter to change:
    PARAM_NAME = "NK_delay"
    PARAM_VALUES = [99_99999.0 for _ in range(10)] + [NK_delay_val for _ in range(10)]

    NK_impact = True

    os.makedirs(f"results_{idx}", exist_ok = True)
    #print(f"Simluations to make: {len(PARAM_VALUES)}")

    with open(f"parameters_{idx}.txt", "w") as f:
        print("----- Parameters -----", file = f)
        for key, val in BASE_params.items():            
            print(f"{key}: {val}", file = f)
    


    for i, v in enumerate(PARAM_VALUES):
        params = BASE_params.copy()
        params[PARAM_NAME] = v

        if NK_impact:
            if params["NK_delay"] > params["time_max"]:
                out_path = f"results_{idx}/without NK, {i}.npz"
            else:
                out_path = f"results_{idx}/with NK, {i}.npz"

        print(f"[{i+1}/{len(PARAM_VALUES)}]; starting: {PARAM_NAME}={v}")
        result = run_with_params(params)
        print(f"done, t = {result["total_time"]}, steps = {result["total_steps"]}")

        np.savez_compressed(out_path, 
                            stats_array = result["stats_array"],
                            frames_NK = result["frames_NK"],
                            frames_epith = result["frames_epith"],
                            **{k: np.array(v) for k, v in params.items()})
        

    print(f"All done for realization {idx}")