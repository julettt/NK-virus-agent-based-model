import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri

grid_size = 50
NK_ratio = 1.0
max_NK = 3
a = 3/4
t_max = 50

#P = a / (1 + b * #NK)

def NK_move_prop(a, b, NK_num):
    return a * NK_num / (1 + b * NK_num)


def run_test(b):
    grid = np.zeros((grid_size, grid_size))

    #NK_grid initial conditions
    NK_grid = np.zeros(grid.shape)
    NK_grid = np.random.poisson(NK_ratio, size = (grid_size, grid_size))
    NK_grid = np.where(NK_grid > 3, 3, NK_grid)

    init_3 = np.sum(np.where(NK_grid == 3, 1, 0))
    init_2 = np.sum(np.where(NK_grid == 2, 1, 0))
    init_1 = np.sum(np.where(NK_grid == 1, 1, 0))
    init_0 = np.sum(np.where(NK_grid == 0, 1, 0))

    NK_moves_propens = np.where(NK_grid > 0, NK_move_prop(a, b, NK_grid), 0) #NK initial move propensities


    def NK_move(r, c):
        #possible directions are different for even and odd rows
        if r % 2 == 0:
            directions = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
        else:
            directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]]
    
        dr, dc = directions[np.random.randint(0, 6)]
        return (r + dr) % grid_size, (c + dc) % grid_size

    time = 0
    steps = 0

    while time < t_max:

        all_propens = NK_moves_propens.flatten() #np.concatenate(zdarzenia_A, zdarzenia_B)
        total_propensity = all_propens.sum()

        #print(all_propens.flatten())
        
        if total_propensity == 0:
            print('total_propensity = 0')
            break

        event_type = ['NK_move' for _ in all_propens]

        #losujemy czas i zdarzenie dla tego czasu
        dt = - np.log(np.random.random()) / total_propensity
        event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)
        
        time += dt
        steps += 1


        if event_type[event_id] == 'NK_move':

            r, c = divmod(event_id, grid_size)
            
            new_r, new_c = NK_move(r, c) #współrzędne komórki która ma się ruszyć

            #ZMIANA: dodanie warunku sprawdzającego, że nie ruszymy się w miejsce gdzie są 3 komórki
            move = False
            while move == False:
                if NK_grid[new_r, new_c] <= 2:
                    NK_grid[r, c] -= 1
                    NK_grid[new_r, new_c] += 1
                    move = True
                else:
                    new_r, new_c = NK_move(r, c)


            #aktualizacja propensities -> robimy ją tylko dla tych dwóch punktów w których się coś zmieniło
            NK_moves_propens[r, c] = NK_move_prop(a, b, NK_grid[r, c]) if NK_grid[r, c] else 0
            NK_moves_propens[new_r, new_c] = NK_move_prop(a, b, NK_grid[new_r, new_c])


    new_3 = np.sum(np.where(NK_grid == 3, 1, 0))
    new_2 = np.sum(np.where(NK_grid == 2, 1, 0))
    new_1 = np.sum(np.where(NK_grid == 1, 1, 0))
    new_0 = np.sum(np.where(NK_grid == 0, 1, 0))
    
    print('-------------')
    print(f'compleated for b = {b}')
    print('total time: ', time, 'steps: ', {steps})
    print(f'initial: 0: {init_0}, 1: {init_1}, 2: {init_2}, 3: {init_3}. Total: {init_0 + init_1 + init_2 + init_3}')
    print(f'after simulation: 0: {new_0}, 1: {new_1}, 2: {new_2}, 3: {new_3}')
    print(f'difference (new - old) 0: {new_0 - init_0}, 1: {new_1 - init_1}, 2: {new_2 - init_2}, 3: {new_3 - init_3}. Total = {new_0 + new_1 + new_2 + new_3}')
    
    
    return NK_grid.flatten()
b_values = [0, 1/3, 1, 3, 10]
results = {b: run_test(b) for b in b_values}


'''
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.1

x = np.arange(4)


for i, b in enumerate(b_values):
    counts = [np.sum(results[b] == val) for val in range(4)]
    ax.bar(x + i * width, [c/np.sum(counts) for c in counts], width, label=f'b={b}')


ax.set_xticks(x)
ax.set_xticklabels(['0 NK', '1 NK', '2 NK', '3 NK'])
ax.set_title(f'For different b, P =  a * #NK/(1 + b * #NK)')
ax.legend()
plt.show()

'''