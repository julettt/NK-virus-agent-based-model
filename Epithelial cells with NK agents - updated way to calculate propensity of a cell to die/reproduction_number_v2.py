import numpy as np

def calculate_analytical_R(p, with_NK = True):
    "function to calculate theoretical reproduction number (R)"
    M_I = p["M_I"]
    M_D = p["M_D"]
    c_0 = p["c_0"]
    c_I = p["c_I"]
    c_NK = p["c_NK"]
    c_I_NK = p["c_I_NK"]
    t_prog = p["t_prog"]
    beta_spread = p["beta_spread"]
    t_spread = p["t_spread"]
    max_inf_state = p["max_inf_state"]

    NK = p["NK_ratio"] if with_NK else 0.0
    R_total = 0.0
    P_reach = 1.0 #probability that a cell will live till state S

    for S in range(1, max_inf_state + 1):
        
        #propensity of NK-independent death
        if S > M_D:
            r_death_ind = c_0 + c_I * S
        else:
            r_death_ind = c_0

        #propensity of NK-dependent death
        if S > M_D:
            r_death_dep = c_NK * NK + c_I_NK * S * NK
        else:
            r_death_dep = c_NK * NK

        r_death = r_death_ind + r_death_dep

        #propensity to progress from state S to state S+1 -> if we reach max_inf_state it is 0
        r_prog = 1.0 / t_prog if S < max_inf_state else 0.0

        #total propensity to leave state S
        r_exit = r_prog + r_death
        if r_exit == 0:
            break

        #mean time spend in state S
        tau = 1.0 / r_exit

        #propensity to infect neighbors:
        if S > M_I:
            lambda_inf = 6.0 * (beta_spread * S) / t_spread
        else:
            lambda_inf = 0.0

        #reproduction number update
        R_total += P_reach * lambda_inf * tau
        
        #probability of progressing to next state S+1
        P_reach *= (r_prog / r_exit)

    return R_total


p = dict(time_max = 10_000.0, virus_delay = 0.0, NK_delay = 4.0, max_steps = 50_000_000, ani_step_save = 5_000,
                   grid_r = 50, grid_c = 500, max_NK = 3, a = 1.0, b = 3.0, max_inf_state = 15,
                   t_prog = 1, beta_spread = 1, t_spread = 50, M_I = 5, M_D = 5,
                   c_0 = 0, c_I = 0.01, c_NK = 0, c_I_NK = 0.03, 
                   MOI = 1, NK_ratio = 1.0, 
                   with_death_I = True, with_death_NK = True, with_infection_spread = True, with_infection_progression = True)



rep_n_with = calculate_analytical_R(p, with_NK = True)
rep_n_without = calculate_analytical_R(p, with_NK = False)


print('-- reproduction number --')
print(f'with NK: {rep_n_with:.3f}')
print(f'without NK: {rep_n_without:.3f}')