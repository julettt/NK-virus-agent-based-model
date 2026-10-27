import random
import csv

N_PARAMS = 10

def simple_linear_equations(params):

    x = [params[f"x{i}"] for i in range(1, N_PARAMS + 1)]

    # e1, ..., e10: e1 = x1 + ... + x10, e2 = x1 + ... + x9, ..., e10 = x1
    e = [sum(x[:k]) for k in range(N_PARAMS, 0, -1)]

    # e11, ..., e18: dodajemy sąsiednie pary, tj. e11 = x1+x2, e12 = x2+x3, ..., e18 = x9+x10
    e += [x[i] + x[i + 1] for i in range(1, N_PARAMS - 1)]

    # e19 i e20: suma parametrów nieparzystych i parzystych
    e.append(sum(x[0::2]))  # x1 + x3 + x5 + x7 + x9
    e.append(sum(x[1::2]))  # x2 + x4 + x6 + x8 + x10

    return e
