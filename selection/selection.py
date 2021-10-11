import numpy as np


def uniform(population, idx, num=3):
        '''
        This function selects num unique parents from the population. It also ensures that none of the parents selected
        is the agent in the population located at idx.

        @param: population (m by n numpy array) - collection of m agents of dimension n
        @param: idx (int) - index of agent not to include in random selection
        @param: num (int) - number of random selections (default 3 as required by differential evolution)
        '''

        # dummy population which does not contain agent at idx
        dummy_pop = np.delete(population, idx, axis=0)

        # Select three unique random agents from dummy pop (uniform random)
        idxes = np.random.choice(len(dummy_pop), size=num, replace=False)

        return [dummy_pop[i] for i in idxes]


def elitist(population, fitness, idx, offspring, f):
    '''
    This function performs deterministic elitist selection between a new offspring with function evaluation f with an 
    existing agent in the population at idx. If the offspring is an improvement, it replaces the old agent, otherwise no
    change is made.

    @param: population (m by n numpy array) - collection of m agents of dimension n
    @param: fitness (m by 1 numpy array) - corresponding fitness values of each agent in population
    @param: idx (int) - index of agent in population being compared
    @param: offspring (n x 1 numpy array) - new candidate offspring
    @param: f (float) - function value of offspring
    '''

    if (f <= fitness[idx]):
        population[idx] = offspring
        fitness[idx] = f


