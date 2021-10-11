import numpy as np

def uniform(parent1, parent2, p=0.5):

    # Sanity check
    assert(len(parent1)==len(parent2))

    # Initialize offspring
    offspring = 0 * parent1

    for i in range(len(parent1)):
        
        # Take allele from parent1 
        if (np.random.uniform() < p):
            offspring[i] = parent1[i]

        # Take allele from parent2
        else:
            offspring[i] = parent2[i]

    return offspring
