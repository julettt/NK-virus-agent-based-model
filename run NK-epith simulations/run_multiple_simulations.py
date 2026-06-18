import numpy as np
import os
from nk_epith_model_v2 import run_with_params
from single_run_with_different_params import run_for_params

#BASE_params = dict(time_max = 10_000.0, virus_delay = 0.0, NK_delay = NK_delay_val, max_steps = 50_000_000, ani_step_save = 10_000,
#                    grid_r = 50, grid_c = 200, max_NK = 3, a = 1.0, b = 3.0, max_inf_state = 15,
#                    t_prog = 1, beta_spread = 1, t_spread = t_spread_val, M_I = 5, M_D = 5,
#                    c_0 = 0, c_I = 0.03, c_NK = 0, c_I_NK = c_I_NK_val, 
#                    MOI = 1, NK_ratio = NK_ratio_val, 
#                    with_death_I = True, with_death_NK = True, with_infection_spread = True, with_infection_progression = True)


T_SPREAD = [50.0, 33.33333, 25.0]

print('-------- t spread --------')
for i, val in enumerate(T_SPREAD):    
    run_for_params(idx = i, t_spread_val = val, c_I_NK_val = 0, NK_ratio_val = 0.1, NK_delay_val = 0.5)

C_I_NK = [0, 0.01, 0.02, 0.03, 0.1]

print('-------- c_I_NK --------')
for i, val in enumerate(C_I_NK):    
    run_for_params(idx = i + len(T_SPREAD), t_spread_val = 25.0, c_I_NK_val = val, NK_ratio_val = 0.1, NK_delay_val = 0.5)

NK_RATIO = [0.1, 0.3, 0.5, 1.0, 1.5]

print('-------- NK_ratio --------')
for i, val in enumerate(C_I_NK):    
    run_for_params(idx = i + len(T_SPREAD) + len(C_I_NK), t_spread_val = 25.0, c_I_NK_val = 0.1, NK_ratio_val = val, NK_delay_val = 0.5)
