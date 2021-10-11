from crossover import crossover


def differential(agents, rate):

    # Base vector 
    base = agents[0]

    # Perturbation vector 
    p = rate * (agents[1] - agents[2])

    # Mutant vector
    M = base + p

    # Trial vector (offspring)
    T = crossover.uniform(base, M)

    return T