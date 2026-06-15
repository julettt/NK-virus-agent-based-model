import numpy as np
import optuna
from nk_epith_model import run_with_params

def calculate_analytical_R(p, with_NK = True):
    "function to calculate theoretical reproduction number (R)"
    M_I_spread = p["M_I_spread"]
    M_I_death = p["M_I_death"]
    gamma_ind = p["gamma_ind"]
    t_ind = p["t_ind"]
    gamma_dep = p["gamma_dep"]
    t_dep = p["t_dep"]
    t_prog = p["t_prog"]
    beta_spread = p["beta_spread"]
    t_spread = p["t_spread"]
    max_inf_state = p["max_inf_state"]

    NK = p["NK_ratio"] if with_NK else 0.0
    R_total = 0.0
    P_reach = 1.0 #probability that a cell will live till state S

    for S in range(1, max_inf_state + 1):
        
        #propensity of NK-independent death
        if S > M_I_death:
            r_death_ind = (gamma_ind + S) / t_ind
        else:
            r_death_ind = gamma_ind / t_ind

        #propensity of NK-dependent death
        if S > M_I_death:
            r_death_dep = (gamma_dep + S) * NK / t_dep
        else:
            r_death_dep = gamma_dep * NK / t_dep

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
        if S > M_I_spread:
            lambda_inf = 6.0 * (beta_spread * S) / t_spread
        else:
            lambda_inf = 0.0

        #reproduction number update
        R_total += P_reach * lambda_inf * tau
        
        #probability of progressing to next state S+1
        P_reach *= (r_prog / r_exit)

    return R_total


def objective(trial):
    p = {
        "time_max": 120.0,
        "virus_delay": 0.0,
  
        "max_steps": 1_000_000,
        "ani_step_save": 10_000,
        "grid_size": 50, 
        "max_NK": 3, 
        "a": 1.0, 
        "b": 3.0, 
        "max_inf_state": 30,
        "MOI": 0.1, 
        "NK_ratio": 1.0, 

        "t_prog": 1.0, 
        "beta_spread": 1.0, 
        "t_spread": 100.0, 

        "with_death_dep": True, 
        "with_death_ind": True, 
        "with_infection_spread": True, 
        "with_infection_progression": True,

        "gamma_ind": 0.0514,
        "t_ind": 399.1724,
        "gamma_dep": 399.1724,
        "t_dep": 21.3922,

        #parameters to optimize
        "M_I_spread": trial.suggest_int("M_I_spread", 3, 20),
        "M_I_death": trial.suggest_int("M_I_death", 3, 20),
        "beta_spread": trial.suggest_float("beta_spread", 0.75, 1.25), 
        "t_spread": trial.suggest_float("t_spread", 75.0, 125.0), 

        #"gamma_ind": trial.suggest_float("gamma_ind", 0.05, 3.0), 
        #"t_ind": trial.suggest_float("t_ind", 100.0, 400.0),
        #"gamma_dep": trial.suggest_float("gamma_dep", 0.05, 1.5), 
        #"t_dep": trial.suggest_float("t_dep", 15.0, 150.0),         
        }
    
    R_without_NK = calculate_analytical_R(p, with_NK = False)
    R_with_NK = calculate_analytical_R(p, with_NK = True)
    
    math_penalty = 0.0

    if R_without_NK <= 1.0: 
        #we want to have virus spread and kill cells without NK cells, so we want R > 1.0
        math_penalty += (1.05 - R_without_NK) * 5000.0

    if R_with_NK >= 1.0:
        #we want to stop virus spread when we have NK cells, so we want R < 1.0
        math_penalty += (R_with_NK - 0.95) * 5000.0

    if math_penalty > 0.0:
        #parameters are not prospering, so we don't want them
        return 5e6 + math_penalty
    


    #taking N realizations of the model to average the values
    N_rep = 8
    res_with_NK = []
    res_without_NK = []

    for _ in range(N_rep):
            
        #with NK
        p["NK_delay"] = 4.0
        res_with = run_with_params(p)
        res_with_NK.append(res_with)

        #without NK
        p["NK_delay"] = 99999.0
        res_without = run_with_params(p)
        res_without_NK.append(res_without)


    #mean values for cell states
    mean_healthy_with = np.mean([res["stats_array"][-1, 0] for res in res_with_NK])
    mean_infected_with = np.mean([res["stats_array"][-1, 1] for res in res_with_NK])
    mean_dead_with = np.mean([res["stats_array"][-1, 2] for res in res_with_NK])

    mean_dead_without = np.mean([res["stats_array"][-1, 2] for res in res_without_NK])


    #CONDITIONS BOUNDARIES:
    healthy_bound = 0.3 #cond2
    inf_bound = 0.2 #cond3

    
    total_cells = p["grid_size"] ** 2

    total_penalty = 0.0

    #CONDITION 1: 
    #more deaths with NK than without NK ---> punishment, because NK should help
    if mean_dead_with >= mean_dead_without:
        #very strict penalty for every dead
        total_penalty += (mean_dead_with - mean_dead_without) * 500.0
    
    #CONDITION 2:
    #we want NK to help to minimize the infection spread and help healthy cells survive, therefore 
    #we give a punishment if less that healthy_bond [%] cells are healthy
    healthy_treshold = healthy_bound * total_cells
    if mean_healthy_with < healthy_treshold:
        total_penalty += (healthy_treshold - mean_healthy_with) * 100.0
    
    #CONDITION 3: 
    #we want to lower the infection, so if more that inf_bond [%] cells are infected we give a punishment
    inf_treshold = inf_bound * total_cells
    if mean_infected_with > inf_treshold:
        total_penalty += (mean_infected_with - inf_treshold) * 100.0
    

    #parameter normalization
    #we want high gamma_dep and low t_dep to have as aggressive NK as possible
    norm_gamma_dep = (p["gamma_dep"] - 0.001) / (1.5 - 0.001)
    norm_t_dep = (150.0 - p["t_dep"]) / (150.0 - 15.0)
    
    #we want low NK-independent death, so we prefer low gamma_ind and high t_ind
    norm_gamma_ind = (3.0 - p["gamma_ind"]) / (3.0 - 0.05) #1 -> max considered agression
    norm_t_ind = (p["t_ind"] - 100.0) / (400.0 - 100.0)
        

    #prize for aggressive and fast-acting NK
    #bigger prize for low healthy-cells mortality
    score = (mean_dead_with / total_cells * 100) + total_penalty - (20 * norm_gamma_dep) - (20 * norm_t_dep) - (30 * norm_gamma_ind) - (30 * norm_t_ind)
    return score

if __name__ == "__main__":

    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 400, n_jobs = -1)
    
    print("best parameters:")
    for key, value in study.best_params.items():
        print(f"{key} = {value:.4f}")

