"""
NK-Virus Agent-Based Model

This model aims to simulate viral infection spread on a hexagonal epithelial cell grid and the surveillance response by NK (Natural Killer) cells.

----- MODEL OVERVIEW -----
The tissue is represented as a 2D grid of epithelial cells arranged on a hexagonal grid where each cell has 6 neighbors. Each epithelial cell has:

    - a *state* s in {0, 1, ..., s_max}: an integer standing in for how far along its infection has progressed where 0 = healthy cell, s_max = fully infected 
      cell with maximal viral protein load. A cell that has died is marked with s = -1 and its status is changed to 'not alive' (see `epith_is_alive`).
    - a boolean *alive* flag.

On the top of the same grid, each node also holds a non-negative integer number of NK cells, which can migrate between neighboring nodes and increase cell's 
chance of dying.

Four types of stochastic events can occur at any grid node:
    1. NK cell migration to a neighboring node,
    2. infection progression (s -> s + 1) in an already-infected epithelial cell,
    3. infection spread from an infectious epithelial cell to a healthy neighbor,
    4. cell death, either "spontaneous" (age-related), induced by viral infection or a presence of NK cell(s).

The simulation advances with the Gillespie stochastic simulation algorithm: at every step the propensity of every possible event at every node is known and the 
time to the next event is drawn from an exponential distribution with rate equal to the sum of all propensities. Then the specific event that occurs is chosen
with probability proportional to its propensity.

----- RUNNING A SIMULATION -----
The typical entry point is `run_with_params(p)` which takes a dictionary of parameters, builds the initial grids and calls the `run_simulation` function
that implements the actual Gillespie loop. 
"""


import numpy as np
import scipy.stats as stats
from numba import njit


# ----- MODULE-LEVEL CONSTANTS -----
#    - `IDX_NK_MOVE`, `IDX_INF_EVOL`, `IDX_INF_SPREAD`, `IDX_DEATH`: indices identifying the four event types listed above inside the propensity array 
#       used by `run_simulation`
#    - `FRAME_CHECKPOINTS`: time points (in hours) at which a frame is always saved, regardless of the regular `ani_step_save` interval. Useful to record 
#       biologically relevant times (e.g. 24h/48h/72h post-infection)

IDX_NK_MOVE = 0
IDX_INF_EVOL = 1
IDX_INF_SPREAD = 2
IDX_DEATH = 3
N_EVENT_TYPES = 4   #total number of possible events

FRAME_CHECKPOINTS = (24.0, 48.0, 72.0)


#------------------------------------------------------------
#               GRID INITIALIZATION FUNCTIONS
#------------------------------------------------------------

def init_epith_grid(moi, grid_r, grid_c):
    """Build the initial epithelial cell state grid.

    Each cell is independently infected (state s = 1) with probability derived from the multiplicity of infection (MOI) under a Poisson model of viral
    exposure: P(infected) = 1 - e^(-MOI).

    Args:
        moi: multiplicity of infection (average number of virions per cell).
        grid_r: number of grid rows.
        grid_c: number of grid columns.

    Returns:
    An (grid_r, grid_c) int64 array with entries 0 (= healthy cell) or 1 (= just-infected cell, i.e. state s = 1).  
    """

    p_inf = 1 - np.exp(-moi)
    epith_grid = (np.random.random((grid_r, grid_c)) < p_inf).astype(np.int64)
    
    return epith_grid


def init_nk_grid(nk_ratio, max_nk, grid_r, grid_c):
    """Build the initial NK cell count grid.

    The number of NK cells at each node is drawn independently from a Poisson distribution with mean `nk_ratio` (the NK-to-epithelial-cell ratio), 
    truncated to at most `max_nk` cells per node and renormalized so that the truncated distribution still sums to 1.
    
    Args:
        nk_ratio: mean number of NK cells per grid node (Poisson mean).
        max_nk: maximum number of NK cells allowed at a single node.
        grid_r: number of grid rows.
        grid_c: number of grid columns.
 
    Returns:
        An (grid_r, grid_c) array of integers in [0, max_nk], that represents the initial NK-cell count at each node.
    """

    k_values = np.arange(max_nk + 1)
    probs = stats.poisson.pmf(k_values, mu = nk_ratio)
    probs /= probs.sum()    #renormalize after truncating the Poisson tail at max_nk

    nk_grid = np.random.choice(k_values, size = (grid_r, grid_c), p = probs)

    return nk_grid


@njit(cache=True)
def get_hex_neighbors(r, c, grid_r, grid_c):
    """Return the six hexagonal-grid neighbors of node (r, c).
 
    The grid is stored as a rectangular array but represents a hexagonal grid using the common "offset coordinates" scheme: even and odd rows
    are horizontally shifted by half a cell relative to each other, so the six neighbor directions differ depending on the parity of the row `r`.
 
    Boundary conditions: the grid always wraps around (periodic boundary conditions).
 
    Args:
        r: row index of the node.
        c: column index of the node.
        grid_r: total number of grid rows.
        grid_c: total number of grid columns.
 
    Returns:
        A (6, 2) int64 array where row i holds the (row, col) coordinates of the i-th neighbor.
    """
    
    if r % 2 == 0:
        directions = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))
    else:
        directions = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))

    result = np.empty((6, 2), dtype = np.int64)
    
    for i in range(6):

        result[i, 0] = (r + directions[i][0]) % grid_r
        result[i, 1] = (c + directions[i][1]) % grid_c

    return result


#------------------------------------------------------------
#                   FENWICK TREE UTILITIES
#------------------------------------------------------------

@njit(cache=True)
def fenwick_build(flat_propensity, n):
    """Build a Fenwick tree (1-indexed, size n+1) from a flat 0-indexed propensity array.

    Args:
        flat_propensity: (n,) array of non-negative propensities, one per flattened grid node (node (r, c) maps to index r * grid_c + c).
        n: number of nodes (grid_r * grid_c).

    Returns:
        A (n + 1,) float64 Fenwick tree encoding prefix sums of `flat_propensity`.
    """
    tree = np.zeros(n + 1, dtype=np.float64)

    for i in range(n):
        idx = i + 1
        tree[idx] += flat_propensity[i]
        j = idx + (idx & (-idx))

        if j <= n:
            tree[j] += tree[idx]

    return tree

@njit(cache=True)
def fenwick_update(tree, i, delta, n):
    """Add `delta` to the propensity at flat index `i` (0-indexed).

    Must be called with delta = new_value - old_value any time the propensity at node `i` changes.

    Args:
        tree: (n + 1,) Fenwick tree as returned by `fenwick_build`, modified in place.
        i: 0-indexed flat node index (r * grid_c + c) whose propensity changed.
        delta: signed change to apply (new_value - old_value).
        n: number of nodes (grid_r * grid_c).
    """

    idx = i + 1

    while idx <= n:
        tree[idx] += delta
        idx += idx & (-idx)


@njit(cache=True)
def fenwick_find(tree, rand_val, n, log_n):
    """Find the flat index whose cumulative propensity contains `rand_val`.

    Given a Fenwick tree of non-negative propensities and a target value in [0, total propensity), returns the 0-indexed flat node (r * grid_c + c)
    at which the running cumulative sum first reaches `rand_val`.

    Args:
        tree: (n + 1,) Fenwick tree as returned by `fenwick_build`/kept in sync by `fenwick_update`.
        rand_val: target value in [0, total propensity); caller draws this as (uniform random number * total propensity of this event type).
        n: number of nodes (grid_r * grid_c).
        log_n: highest power of two <= n (precompute once per grid size, outside the hot loop; it never changes during a run, so there is no need
        to recompute it on every event).

    Returns:
        The 0-indexed flat node index; convert back to (r, c) via r = idx // grid_c, c = idx % grid_c.
    """

    idx = 0
    remaining = rand_val
    pw = log_n

    while pw > 0:
        next_idx = idx + pw

        if next_idx <= n and tree[next_idx] <= remaining:
            idx = next_idx
            remaining -= tree[next_idx]

        pw >>= 1

    if idx >= n:
        return n - 1

    return idx


#------------------------------------------------------------
#                   PROPENSITY FUNCTIONS
#------------------------------------------------------------

@njit(cache=True)
def progression_prop(t_prog, r_val,  with_infection_progression):
    """Propensity of an infected cell to progress from state s to s + 1.
 
    This propensity does not depend on which cell or its neighbors are in which state (it is a constant rate, r_val/t_prog, applied to any cell
    with 0 < s < s_max) - the caller is responsible for only invoking this for such cells.
 
    Args:
        t_prog: mean time (in hours) for a single s -> s + 1 transition.
        r_val: reduction of infectivity by NK in WT cells.
        with_infection_progression: if False, infection progression is disabled and this always returns 0.
 
    Returns:
        The propensity of the s -> s + 1 transition.
    """

    if not with_infection_progression:
        return 0.0

    return r_val / t_prog


@njit(cache=True)
def infect_prop(r, c, epith_grid, epith_is_alive, grid_r, grid_c, t_spread, m_i, with_infection_spread, s_max):
    """Propensity of a healthy cell at (r, c) to become infected.
 
    This must be called for a healthy, living cell (its own state is not checked here). It sums a contribution from each living neighbor whose
    infection state s exceeds the infectiousness threshold `m_i`: a more heavily infected neighbor (s closer to `s_max`) contributes a
    proportionally larger rate.
 
    Args:
        r, c: coordinates of the (healthy) cell being evaluated.
        epith_grid: (grid_r, grid_c) array of epithelial-cell states s.
        epith_is_alive: (grid_r, grid_c) boolean array, cell survival status.
        grid_r, grid_c: grid dimensions.
        t_spread: characteristic time scale of infection spread.
        m_i: state threshold above which a neighbor is infectious.
        with_infection_spread: if False, this mechanism is disabled and the function always returns 0.
        s_max: maximum possible value of the infection state s, used to normalize a neighbor's contribution to [0, 1].
 
    Returns:
        The total propensity of (r, c) becoming infected summed over all its infectious living neighbors.
    """

    if not with_infection_spread:
        return 0.0
    
    total_inf_prop = 0.0

    neighbors = get_hex_neighbors(r, c, grid_r, grid_c)

    for i in range(6):
        nr, nc = neighbors[i, 0], neighbors[i, 1]
        
        if not epith_is_alive[nr, nc]: #dead neighboring cell cannot infect other cells
            continue

        s_neigh = epith_grid[nr, nc]

        if s_neigh > m_i:
            s_norm = s_neigh / s_max
            total_inf_prop += s_norm / t_spread
    
    return total_inf_prop


@njit(cache=True)
def death_independent_prop(r, c, epith_grid, c_0, c_i, m_npf, with_death_independent, s_max):
    """Propensity of a cell at (r, c) to die independently of its neighbors.
 
    Note: The pre-introduction period [0, min(virus_delay, nk_delay))
    is an unmodeled warm-up phase intended to preserve perfect tissue integrity prior to infection; baseline turnover begins exactly when the first
    external factor (virus or NK cells presence) is introduced.

    Args:
        r, c: coordinates of the cell being evaluated.
        epith_grid: (grid_r, grid_c) array of epithelial-cell states s.
        c_0: baseline death rate applying to every living cell.
        c_i: extra death-rate coefficient for heavily infected cells.
        m_npf: structural protein production threshold (NP for IAV, F for RSV).
        with_death_independent: if False, this mechanism is disabled and the function always returns 0.
        s_max: maximum possible value of s.
 
    Returns:
        The death propensity for this cell, ignoring any NK-induced contribution (see `death_nk_prop`).
    """

    if not with_death_independent:
        return 0.0

    s_cell = epith_grid[r, c]

    if s_cell > m_npf:
        s_norm = s_cell / s_max
        return c_0 + c_i * s_norm

    return c_0


@njit(cache=True)
def death_nk_prop(r, c, epith_grid, nk_grid, c_nk, c_i_nk, m_ns1, with_death_nk, s_max):
    """Propensity of a cell at (r, c) to die due to the NK cells present there.
 
    The rate scales linearly with the number of NK cells co-located at the same node. As with `death_independent_prop`, a heavily infected cell
     is killed by NK cells at a higher per-NK-cell rate than a cell that is healthy or only mildly infected.
 
    Args:
        r, c: coordinates of the cell being evaluated.
        epith_grid: (grid_r, grid_c) array of epithelial-cell states s.
        nk_grid: (grid_r, grid_c) array of NK-cell counts.
        c_nk: baseline per-NK-cell killing rate.
        c_i_nk: extra per-NK-cell killing-rate coefficient for heavily infected cells.
        m_ns1: non-structural protein production threshold.
        with_death_nk: if False, this mechanism is disabled and the function always returns 0.
        s_max: maximum possible value of s.
 
    Returns:
        The NK-induced death propensity for this cell.
    """
    
    if not with_death_nk:
        return 0.0

    nk_cells = nk_grid[r, c]
    s_cell = epith_grid[r, c]

    if s_cell > m_ns1:
        s_norm = s_cell / s_max

        return nk_cells * (c_nk + c_i_nk * s_norm)

    return nk_cells * c_nk


@njit(cache=True)
def nk_move_prop(r, c, nk_grid, max_nk, grid_r, grid_c, a, b):
    """Propensity of an NK-cell migration event starting from node (r, c).
 
    A migration event represents one NK cell leaving (r, c) for one of its neighboring nodes that still has room (fewer than `max_nk` NK cells). 
    The rate saturates as the local NK-cell density (own node plus its 6 neighbors) grows, via the two-parameter function
 
        rate(r, c) = a * N(r, c) / (a + b * N_local),
 
    where `N(r, c)` is the NK-cell count at (r, c) and `N_local` is the total NK-cell count in the neighborhood (without the cell that is moving).
 
    Args:
        r, c: coordinates of the node NK cells may migrate away from.
        nk_grid: (grid_r, grid_c) array of NK-cell counts.
        max_nk: maximum number of NK cells allowed at any single node.
        grid_r, grid_c: grid dimensions.
        a: base migration-rate parameter.
        b: crowding parameter.
 
    Returns:
        0.0 if there are no NK cells at (r, c) or none of its neighbors has free capacity; otherwise the migration propensity for a single NK cell to leave (r, c).
    """

    nk_count = nk_grid[r, c]

    if nk_count == 0:
        return 0.0
    
    neighs = get_hex_neighbors(r, c, grid_r, grid_c)

    move_possible = False
    n_local = nk_count - 1   #the NK cell that is moving is not a neighbor to itself

    for i in range(6):

        n_local += nk_grid[neighs[i, 0], neighs[i, 1]]


        if nk_grid[neighs[i, 0], neighs[i, 1]] < max_nk:
            move_possible = True

    if not move_possible:
        return 0.0
    
    denominator = a + b * float(n_local)

    return (a * float(nk_count)) / denominator
    

#------------------------------------------------------------
#            HELPER FUNCTIONS FOR THE GILLESPIE LOOP
#------------------------------------------------------------

@njit(cache=True)
def run_simulation(time_max, virus_delay, nk_delay, max_steps, ani_step_save, grid_r, grid_c, 
                   max_nk, a, b, s_max, t_prog, r_val, t_spread, m_i, m_npf, m_ns1,
                   c_0, c_i, c_nk, c_i_nk,
                   protein_scale_constant,
                   epith_grid_delayed, empty_grid, nk_grid_delayed,
                   with_death_independent, with_death_nk, with_infection_spread, with_infection_progression):
    """Run the full Gillespie simulation and record snapshots over time.
 
    The tissue starts fully healthy and NK-cell-free. At `virus_delay` hours, the epithelial grid is switched over to `epith_grid_delayed` (the actual 
    initial infection pattern); at `nk_delay` hours, the NK grid is switched over to `nk_grid_delayed`. This models an experiment where the virus 
    and/or the NK cells are introduced into the tissue at a given time rather than being present from time 0. Until its introduction time, each of 
    the two grids is simply `empty_grid`.
 
    From then on, the simulation repeatedly:
        1. computes the total propensity of each of the 4 event types across the whole grid,
        2. draws the waiting time to the next event from an exponential distribution with rate equal to the grand total propensity,
        3. draws which event type and which grid node it happens at (probability proportional to propensity),
        4. applies the event and updates the propensities that changed as a result (update is local, cause we update only the affected node and its neighbors);
    until `time_max` is reached, no more events are possible, the virus has been cleared, or `max_steps` events have been executed (a safety cutoff).
 
    Snapshots of both grids and summary statistics are saved periodically (every `ani_step_save` steps) and, in addition, at the fixed checkpoint 
    times listed in `FRAME_CHECKPOINTS`.
 
    Args:
        time_max: simulated end time, in hours.
        virus_delay: time at which the epithelial grid is switched from `empty_grid` to `epith_grid_delayed` (moment of virus introduction).
        nk_delay: time at which the NK grid is switched from `empty_grid` to `nk_grid_delayed` (moment of NK-cell introduction).
        max_steps: hard cap on the number of Gillespie events, used to guarantee the simulation terminates.
        ani_step_save: number of Gillespie steps between two regular saved frames.
        grid_r, grid_c: grid dimensions.
        max_nk: maximum number of NK cells allowed at a single node.
        a, b: NK-cell migration-rate parameters (see `nk_move_prop`).
        s_max: maximum possible infection state s.
        t_prog: mean time for an s -> s + 1 transition (see `progression_prop`).
        r_val: reduction of infectivity by NK in WT (see `progression_prop`).
        t_spread: infection-spread time scale (see `infect_prop`).
        m_i: infectiousness threshold (see `infect_prop`).
        m_npf: structural protein production threshold.
        m_ns1: non-structural protein production threshold.
        c_0, c_i: independent-death-rate parameters.
        c_nk, c_i_nk: NK-induced death-rate parameters.
        protein_scale_constant: constant used to rescale structural and non-structural protein load (to better fit the experimental data).
        epith_grid_delayed: (grid_r, grid_c) array, the epithelial state grid to switch to at `virus_delay` (typically from `init_epith_grid`).
        empty_grid: (grid_r, grid_c) all-zero array used as the "nothing introduced yet" placeholder for both grids.
        nk_grid_delayed: (grid_r, grid_c) array, the NK-cell grid to switch to at `nk_delay` (typically from `init_nk_grid`).
        with_death_independent, with_death_nk, with_infection_spread, with_infection_progression: booleans switching the corresponding mechanism on/off 
        (see the matching propensity functions above).
 
    Returns:
        A 5-tuple:
            frames_nk: (n_frames, grid_r, grid_c) array of saved NK-cell grids.
            frames_epith: (n_frames, grid_r, grid_c) array of saved epithelial-state grids (with dead cells marked as -1).
            stats_array: (n_frames, 4 + 2 * (s_max + 1)) array where each row holds, in order: healthy-cell count, infected-cell count, dead-cell count, 
            simulated time, a histogram (over infection states) of currently-alive cells, and a histogram of the infection states that cells had *at
                the moment they died* (cumulative over the whole run).
            time: the final simulated time reached.
            steps: the total number of Gillespie events executed.
    """

    # ----- PREPARING GRIDS -----

    nk_grid = empty_grid.copy()
    epith_grid = empty_grid.copy()
    epith_is_alive = np.full((grid_r, grid_c), True, dtype = np.bool_)

    n_cells = grid_r * grid_c
    log_n = 1 << int(np.floor(np.log2(n_cells)))

    #propensities of the 4 possible event types are kept in one combined 3D array (indexed by IDX_NK_MOVE/IDX_INF_EVOL/IDX_INF_SPREAD/IDX_DEATH) together with their
    #running totals (propensity_totals) - see _set_propensity below.

    propensity_grids = np.zeros((N_EVENT_TYPES, grid_r, grid_c), dtype = np.float64)
    propensity_totals = np.zeros(N_EVENT_TYPES, dtype = np.float64)

    fenwick_trees = np.zeros((N_EVENT_TYPES, n_cells + 1), dtype = np.float64)

    #histogram over infection state level (s in [0, ..., s_max]) of cells corresponding to the viral-protein level at the moment they die, accumulated over the 
    #whole simulation
    dead_inf_hist = np.zeros(s_max + 1, dtype = np.float64)

    nk_introduced = False
    virus_introduced = False

    time = 0.0
    prev_time = 0.0
    steps = 0
    active_infections = 0   #number of infected cells at a given time
    prop_total = 0.0

    #pre-allocate storage for the saved frames/statistics; +2 leaves room for the initial frame and one final frame beyond the regular grid + room for the 
    #FRAME_CHECKPOINTS
    max_frames = (max_steps // ani_step_save) + 2 + len(FRAME_CHECKPOINTS)
    frames_nk = np.zeros((max_frames, grid_r, grid_c), dtype = nk_grid.dtype)
    frames_epith = np.zeros((max_frames, grid_r, grid_c), dtype = epith_grid.dtype)
    stats_array = np.zeros((max_frames, 4 + 2 * (s_max + 1) + 2), dtype = np.float64)
    frame_idx = 0


    #----- NESTED HELPER FUNCTIONS -----

    def _set_propensity(event_idx, r, c, val):
        """Overwrite the propensity of one event type at (r, c) with `val`, keeping `propensity_totals[event_idx]` consistent by adjusting it by the 
        difference between the new and the old value."""

        old_val = propensity_grids[event_idx, r, c]
        delta = val - old_val

        propensity_totals[event_idx] += delta
        propensity_grids[event_idx, r, c] = val

        flat_idx = r * grid_c + c
        fenwick_update(fenwick_trees[event_idx], flat_idx, delta, n_cells)

 
    def _update_epith(r, c):
        """Recompute every epithelial-cell-related propensity that depends on the state of (r, c): its own progression/death propensities and the
        infection-spread propensity of any healthy living neighbor (since that depends on (r, c)'s own state).
 
        Must be called any time (r, c)'s state, alive/dead status, or the NK-cell count at (r, c) changes."""
        
        #----- DEAD CELL -----
        #it can no longer progress, die again or spread infection to others, but a neighbor that just died can still affect the infection-spread propensity
        #of its living neighbors (that's one fewer infectious source)


        if not epith_is_alive[r, c]:
            _set_propensity(IDX_INF_EVOL, r, c, 0.0)
            _set_propensity(IDX_DEATH, r, c, 0.0)
            _set_propensity(IDX_INF_SPREAD, r, c, 0.0)

            #updating the neighbors' propensities to get infected, since one of their potential sources of infection just disappeared
            neighbors = get_hex_neighbors(r, c, grid_r, grid_c)
            
            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]

                if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #the neighbor is both alive and healthy, so it can get infected
                    _set_propensity(IDX_INF_SPREAD, nr, nc, 
                                    infect_prop(nr, nc, epith_grid, epith_is_alive, grid_r, grid_c, t_spread, m_i, with_infection_spread, s_max))

            return
        

        #----- ALIVE CELL -----
        s = epith_grid[r, c]

        #for healthy cell (s == 0): cell can be infected, infection can't progress (cause there is no infection), cell can die
        #for infected cell (0 < s <= s_max): cell can't be infected again, infection can progress (not true for s = s_max), cell can die

        _set_propensity(IDX_INF_SPREAD, r, c, 
                        infect_prop(r, c, epith_grid, epith_is_alive, grid_r, grid_c, t_spread, m_i, with_infection_spread, s_max) if s == 0 else 0.0)

        _set_propensity(IDX_INF_EVOL, r, c,
                        progression_prop(t_prog, r_val, with_infection_progression) if 0 < s < s_max else 0.0)
        
        _set_propensity(IDX_DEATH, r, c,
                        death_independent_prop(r, c, epith_grid, c_0, c_i, m_npf, with_death_independent, s_max)
                        + death_nk_prop(r, c, epith_grid, nk_grid, c_nk, c_i_nk, m_ns1, with_death_nk, s_max))


        # this cell's own state may have changed, so its healthy living neighbors' infection-spread propensity needs to be refreshed too
        neighbors = get_hex_neighbors(r, c, grid_r, grid_c)
        for i in range(6):
            nr, nc = neighbors[i, 0], neighbors[i, 1]

            if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #our neighbor is both alive and healthy, so it can get infected
                    _set_propensity(IDX_INF_SPREAD, nr, nc, 
                                    infect_prop(nr, nc, epith_grid, epith_is_alive, grid_r, grid_c, t_spread, m_i, with_infection_spread, s_max))

    def _update_nk(r, c):
        """Recompute the NK-cell migration propensity at (r, c). Must be called any time the NK-cell count at (r, c) or at one of its neighbors changes."""
        _set_propensity(IDX_NK_MOVE, r, c, nk_move_prop(r, c, nk_grid, max_nk, grid_r, grid_c, a, b))


    def _save_frame():
        """Append the current state of both grids, plus summary statistics, as the next saved frame."""

        nonlocal frame_idx

        frames_nk[frame_idx] = nk_grid.copy()
        frames_epith[frame_idx] = epith_grid.copy()

        #----- CELL-COUNT STATISTICS -----
        healthy = 0
        infected = 0
        dead = 0
        for r in range(grid_r):
            for c in range(grid_c):
                if not epith_is_alive[r, c]:
                    dead += 1
                elif epith_grid[r, c] == 0:
                    healthy += 1
                else:
                    infected += 1

        stats_array[frame_idx, 0] = float(healthy)
        stats_array[frame_idx, 1] = float(infected)
        stats_array[frame_idx, 2] = float(dead)
        stats_array[frame_idx, 3] = float(time)

        #histogram of the states of cells in the (living) cell population
        inf_histogram = np.zeros(s_max + 1, dtype = np.float64)

        #mean non-structural (NS1) and structural (NP for IAV, F for ISV) protein level over LIVING (both healthy and infected) cells
        ns1_total = 0.0
        npf_total = 0.0
        n_alive = 0

        for r in range(grid_r):
            for c in range(grid_c):
                s = epith_grid[r, c]
                if epith_is_alive[r, c]:
                    inf_histogram[s] += 1
                    n_alive += 1
                    ns1_total += max(0, s - m_ns1) * protein_scale_constant
                    npf_total += max(0, s - m_npf) * protein_scale_constant               

        if n_alive > 0:
            ns1_mean = ns1_total / n_alive
            npf_mean = npf_total / n_alive
        else:
            ns1_mean = 0.0
            npf_mean = 0.0 

        #row layout: [healthy, infected, dead, time, <state for alive cells histogram>, <state at-death histogram>, mean NS1 per living cell, mean NPF per living cell]
        stats_array[frame_idx, 4: 4 + s_max + 1] = inf_histogram
        stats_array[frame_idx, 4 + s_max + 1 : 4 + 2 * (s_max + 1)] = dead_inf_hist
        stats_array[frame_idx, 4 + 2 * (s_max + 1)] = ns1_mean
        stats_array[frame_idx, 4 + 2 * (s_max + 1) + 1] = npf_mean
           
        frame_idx += 1


    #------------------------------------------------------------
    #                    MAIN GILLESPIE LOOP
    #------------------------------------------------------------

    while time <= time_max:     

        # ----- introducing virus at t = virus_delay -----
        if not virus_introduced and time >= virus_delay:
            epith_grid = epith_grid_delayed.copy()
            for r in range(grid_r):
                for c in range(grid_c):
                    if epith_grid[r, c] > 0:    #counting initially infected cells
                        active_infections += 1
                    _update_epith(r, c)    #propensities must reflect the new grid
                    
            virus_introduced = True


        # ----- introducing NK cells at t = nk_delay -----
        if not nk_introduced and time >= nk_delay:
            nk_grid = nk_grid_delayed.copy()
            for r in range(grid_r):
                for c in range(grid_c):
                    _update_nk(r, c)
                    _update_epith(r, c) #death propensities depend on NK-cell counts too
                    
            nk_introduced = True

        
        # ----- saving frames -----
        #regular frames
        if steps % ani_step_save == 0:
            _save_frame()

        #extra checkpoint frames
        for checkpoint in FRAME_CHECKPOINTS:
            if prev_time < checkpoint <= time:
                true_time = time
                time = checkpoint   #only for this frame's label
                _save_frame()
                time = true_time    #restoring the real clock

        if steps > 0 and steps % 10_000 == 0:     #float drift correction
            for k in range(N_EVENT_TYPES):
                flat_prop = propensity_grids[k].flatten()
                fenwick_trees[k] = fenwick_build(flat_prop, n_cells)
                propensity_totals[k] = np.sum(flat_prop)


        # ----- TOTAL PROPENSITY ACROSS ALL EVENT TYPES AND ALL NODES -----
        prop_total = (propensity_totals[IDX_NK_MOVE] + propensity_totals[IDX_INF_EVOL] + propensity_totals[IDX_INF_SPREAD] + propensity_totals[IDX_DEATH])


        # ----- STOPPING CONDITIONS -----  
        if prop_total <= 0.0:
            #nothing can happen right now; if this is only because the virus/NK cells have not been introduced yet, fast-forward the clock to whichever of the
            #two is due next rather than stopping
            next_delay = time_max + 1.0

            if not virus_introduced and virus_delay <= time_max:
                next_delay = min(next_delay, virus_delay)
            if not nk_introduced and nk_delay <= time_max:
                next_delay = min(next_delay, nk_delay)

            if next_delay <= time_max:
                time = next_delay
                continue

            print('No possible events - end of the simulation.')
            break

        if virus_introduced and active_infections == 0:
            #the virus has been cleared from the tissue (no infected cells left and none can become infected) - nothing interesting can happen from here except
            #further NK-cell migration/death, so stop early instead of running until time_max
            if nk_introduced:
                print('Virus is gone.')
            else:
                #infection was but it disappeared before adding NK cells to the system
                print('Virus is gone with no presence of NK cells in the system.')
            break


        #------------------------------------------------------------
        #                   GILLESPIE DIRECT METHOD
        #------------------------------------------------------------

        r1 = np.random.random()
        r2 = np.random.random()

        #waiting time to the next event ~ exponential(rate = prop_total)
        prev_time = time
        time += -np.log(r1) / prop_total

        #value used to select event type and node; uniform in [0, prop_total)
        rand_val = r2 * prop_total
        
        #executing chosen event: the event types are laid out end-to-end along [0, prop_total):
        #[0, NK_MOVE) | [NK_MOVE, NK_MOVE+INF_EVOL) | ... | [..., prop_total); `rand_val` falling in one of these ranges selects that event type; `select_event`
        #then picks a specific node within that event type, weighted by its own propensity.

        #----- event: NK cell move -----
        if rand_val < propensity_totals[IDX_NK_MOVE]:
            flat_idx = fenwick_find(fenwick_trees[IDX_NK_MOVE], rand_val, n_cells, log_n)
            r = flat_idx // grid_c
            c = flat_idx % grid_c
            neighbors = get_hex_neighbors(r, c, grid_r, grid_c)

            #collect neighbors that still have room for one more NK cell
            valid_r = np.empty(6, dtype = np.int64)
            valid_c = np.empty(6, dtype = np.int64)
            n_valid = 0 #number of valid neighbors

            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]

                if nk_grid[nr, nc] < max_nk:
                    valid_r[n_valid] = nr
                    valid_c[n_valid] = nc
                    n_valid += 1

            if n_valid > 0:
                #move possible -> destination chosen uniformly among the eligible neighbors
                idx = np.random.randint(0, n_valid)
                new_r, new_c = valid_r[idx], valid_c[idx]

                nk_grid[r, c] -= 1
                nk_grid[new_r, new_c] += 1

                #propensity update for NK-cell migration: the source and destination nodes, plus all of their neighbors, may have changed migration 
                #propensity due to the changed local NK-cell density
                _update_nk(r, c)
                nb_source = get_hex_neighbors(r, c, grid_r, grid_c)
                for i in range(6):
                    _update_nk(nb_source[i, 0], nb_source[i, 1])

                _update_nk(new_r, new_c)
                nb_dest = get_hex_neighbors(new_r, new_c, grid_r, grid_c)
                for i in range(6):
                    _update_nk(nb_dest[i, 0], nb_dest[i, 1])

                #propensity update for epithelial cells: only the source and destination nodes' own death propensity depends on their own NK-cell 
                #count, so neighbors don't need updating
                _update_epith(r, c)
                _update_epith(new_r, new_c)
                

        #----- event: infection evolution (s -> s + 1) -----
        elif rand_val < propensity_totals[IDX_NK_MOVE] + propensity_totals[IDX_INF_EVOL]:
            adj_rand_val = rand_val - propensity_totals[IDX_NK_MOVE]
            flat_idx = fenwick_find(fenwick_trees[IDX_INF_EVOL], adj_rand_val, n_cells, log_n)
            r = flat_idx // grid_c
            c = flat_idx % grid_c

            #updating infection state and propensities
            epith_grid[r, c] += 1
            _update_epith(r, c)


        #----- event: infection spread to a healthy neighbor -----
        elif rand_val < (propensity_totals[IDX_NK_MOVE] + propensity_totals[IDX_INF_EVOL] + propensity_totals[IDX_INF_SPREAD]):
            adj_rand_val = rand_val - propensity_totals[IDX_NK_MOVE] - propensity_totals[IDX_INF_EVOL]
 
            flat_idx = fenwick_find(fenwick_trees[IDX_INF_SPREAD], adj_rand_val, n_cells, log_n)    #choosing cell to become infected
            r = flat_idx // grid_c
            c = flat_idx % grid_c

            epith_grid[r, c] = 1  #newly infected cells always start at state S = 1
            active_infections += 1
            _update_epith(r, c)


        #----- event: cell death (last type of event) -----
        else:
            adj_rand_val = (rand_val - propensity_totals[IDX_NK_MOVE] - propensity_totals[IDX_INF_EVOL] - propensity_totals[IDX_INF_SPREAD])

            flat_idx = fenwick_find(fenwick_trees[IDX_DEATH], adj_rand_val, n_cells, log_n)     #selecting cell to kill
            r = flat_idx // grid_c
            c = flat_idx % grid_c

            if epith_grid[r, c] > 0:    #the cell that is about to die was infected
                active_infections -= 1

            #record the state the cell had at the moment of death
            dead_inf_hist[epith_grid[r,c]] += 1

            epith_is_alive[r, c] = False
            epith_grid[r, c] = -1   #sentinel value marking a dead cell
            _update_epith(r, c)

        steps += 1

        if steps >= max_steps:
            print('Maximum number of steps exceeded!')
            break

    _save_frame()   #always record the final state, even if the loop ended with break
    return frames_nk[:frame_idx], frames_epith[:frame_idx], stats_array[:frame_idx], time, steps


#------------------------------------------------------------
#                   SIMULATION ENTRY POINT
#------------------------------------------------------------


def run_with_params(p: dict):
    """Build the initial grids and run one full simulation.
 
    This is the main function meant to be called from outside the module. It wraps `run_simulation` (which needs its many numeric arguments passed positionally, 
    for numba compatibility) with a single parameters dictionary.
 
    Args:
        p: dictionary with the following required keys:
            Grid geometry:
                "grid_r", "grid_c": grid dimensions. Select an even value for the `grid_r` to avoid asymetric at the wrap seam.
            Initial-condition parameters:
                "moi": multiplicity of infection, see `init_epith_grid`.
                "nk_ratio", "max_nk": NK-cell initialization parameters, see `init_nk_grid`.
            Timing:
                "time_max": time after which simulation ends (in hours).
                "virus_delay", "nk_delay": times at which the virus/NK cells are introduced (in hours).
                "max_steps": maximum number of steps in the simulation.
                "ani_step_save": number of steps between saved frames.
            Model/rate parameters:
                "a", "b": NK-cell migration parameters, see `nk_move_prop`.
                "s_max": maximum infection state.
                "t_prog", "t_spread": characteristic time scales, see `progression_prop`, `infect_prop`.
                "r_val": reduction of infectivity.
                "m_i", "m_npf", "m_ns1": infectious/protein state thresholds, see `infect_prop`, `death_independent_prop`, `death_nk_prop`.
                "c_0", "c_i": independent-death-rate parameters (natural death rate and NK_independent virus-induced death-rate).
                "c_nk", "c_i_nk": NK-induced and NK_virus-induced death-rate parameters.
            Mechanism on/off switches (booleans):
                "with_death_independent", "with_death_nk", "with_infection_spread", "with_infection_progression".
 
    Returns:
        A dictionary with keys:
            "stats_array": per-frame summary statistics, see the `stats_array` entry in `run_simulation`'s docstring.
            "frames_nk", "frames_epith": per-frame grid snapshots.
            "total_time": final simulated time reached.
            "total_steps": total number of steps executed.
            "params": the input dictionary `p`, for future reference.
    """

    epith_grid_delayed = init_epith_grid(p["moi"], p["grid_r"], p["grid_c"])
    empty_grid = np.zeros((p["grid_r"], p["grid_c"]), dtype = np.int64)
    nk_grid_delayed = init_nk_grid(p["nk_ratio"], p["max_nk"], p["grid_r"], p["grid_c"])
 
    frames_nk, frames_epith, stats_array, total_time, total_steps = run_simulation(
                                                                    p["time_max"], p["virus_delay"], p["nk_delay"],
                                                                    p["max_steps"], p["ani_step_save"], int(p["grid_r"]), int(p["grid_c"]),
                                                                    int(p["max_nk"]), p["a"], p["b"], int(p["s_max"]),
                                                                    p["t_prog"], p["r_val"], p["t_spread"], int(p["m_i"]), int(p["m_npf"]), int(p["m_ns1"]),
                                                                    p["c_0"], p["c_i"], p["c_nk"], p["c_i_nk"],
                                                                    p["protein_scale_constant"],
                                                                    epith_grid_delayed, empty_grid, nk_grid_delayed,
                                                                    p["with_death_independent"], p["with_death_nk"], p["with_infection_spread"],
                                                                    p["with_infection_progression"])
 
    return {"stats_array": stats_array,
            "frames_nk": frames_nk,
            "frames_epith": frames_epith,
            "total_time": total_time,
            "total_steps": total_steps,
            "params": p}