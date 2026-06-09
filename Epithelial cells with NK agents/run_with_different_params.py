import numpy as np
import os
from nk_epith_model import run_with_params

BASE_params = dict(time_max = 72.0, time_delay = 400.0, max_steps = 1000000, ani_step_save = 250,
                   grid_size = 100, max_NK = 3, a = 1.0, b = 3.0, max_inf_state = 10,
                   t_prog = 1.0, beta_spread = 1.0, t_spread = 50.0, M_I = 4,
                   gamma_ind = 1.0, t_ind = 300.0, gamma_dep = 3.0, t_dep = 100.0,
                   MOI = 0.1, NK_ratio = 3.0)


#parameter to change:
PARAM_NAME = "MOI"
PARAM_VALUES = [0.01, 0.1, 1.0]

os.makedirs("results", exist_ok = True)
print(f"Simluations to make: {len(PARAM_VALUES)}")

for i, v in enumerate(PARAM_VALUES):
    params = BASE_params.copy()
    params[PARAM_NAME] = v

    if params["time_delay"] > params["time_max"]: #no nk
        out_path = f"plot results/without NK, {PARAM_NAME}={v}.npz"
    else:
        out_path = f"plot results/with NK, {PARAM_NAME}={v}.npz"

    """if os.path.exists(out_path):
        print(f"[{i+1}/{len(PARAM_VALUES)}]; skipped (previously done): {PARAM_NAME}={v}")
        continue"""

    print(f"[{i+1}/{len(PARAM_VALUES)}]; starting: {PARAM_NAME}={v}")
    result = run_with_params(params)
    print(f"done, t = {result["total_time"]}, steps = {result["total_steps"]}")

    np.savez_compressed(out_path, 
                        stats_array = result["stats_array"],
                        **{k: np.array(v) for k, v in params.items()})
    
print("All done.")
