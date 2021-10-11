import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from objective import analytic
from selection import selection
from crossover import crossover
from mutation import mutation

#----------------------------------------------------------------------------------------------------------------------#
# Parameters
#----------------------------------------------------------------------------------------------------------------------#
n = 10              # Dimension of optimization problem (num of parameters in objective function)
pop_size = 100      # Population size
min_init = -10      # Minimum value an allele can be initialized with
max_init = 10       # Maximum value an allele can be initialized with
eps = 1e-5          # Convergance tolerance
F = 1               # Scale factor
max_gens = 1000     # Max number of generations


#----------------------------------------------------------------------------------------------------------------------#
# Population (Agents) Initialization
#----------------------------------------------------------------------------------------------------------------------#

# Random init
population = np.array([np.random.uniform(min_init, max_init, size=n) for i in range(pop_size)])

# Function evaluations for eaach
fitness = np.array([analytic.ackley(population[i], n=n) for i in range(pop_size)])

#----------------------------------------------------------------------------------------------------------------------#
# Differential Optimization
#----------------------------------------------------------------------------------------------------------------------#

# Starting point
best = np.amin(fitness)
generation = 0

# Save best solution and population average each generation
best_vec = [best]
ave_vec = [np.average(fitness)]

# Loop until solution within tolerance or max generations
while (best > eps) and (generation < max_gens):

    # Increment generation count
    generation += 1

    # Loop over agents (indexes)
    for i in range(len(population)):

        #--------------------------------------------------------------------------------------------------------------#
        # Agent Selection (random uniform)
        #--------------------------------------------------------------------------------------------------------------#
        parents = selection.uniform(population, i)

        #--------------------------------------------------------------------------------------------------------------#
        # Mutation (differential)
        #--------------------------------------------------------------------------------------------------------------#
        offspring = mutation.differential(parents, F)

        #--------------------------------------------------------------------------------------------------------------#
        # Survival (elitist)
        #--------------------------------------------------------------------------------------------------------------#

        # New function evaluation
        f_offspring = analytic.ackley(offspring, n=n)

        # Apply deterministic elite selection between current agent and new offspring 
        selection.elitist(population, fitness, i, offspring, f_offspring)
   

    #------------------------------------------------------------------------------------------------------------------#
    # End of generation
    #------------------------------------------------------------------------------------------------------------------#

    # Print some stats
    print('Generation: {:4d} \tPopulation Average: {:7.4f} \tBest Function Value: {:.5f}'.format(generation, 
                                                                                                 np.average(fitness), 
                                                                                                 best))   
    # Check for solution
    best = np.amin(fitness)

    # Update lists of best / averages
    best_vec.append(best)
    ave_vec.append(np.average(fitness))


#----------------------------------------------------------------------------------------------------------------------#
# Outputs
#----------------------------------------------------------------------------------------------------------------------#

# Solution print
print('Solution: {:s}'.format(np.array2string(population[np.argmin(fitness)])))

# Algorithm performance
plt.figure(1)
plt.plot(np.arange(generation+1), ave_vec, label='population average')
plt.plot(np.arange(generation+1), best_vec, label='best solution')
plt.legend(fontsize=14)
plt.xlabel('Generation', fontsize=14)
plt.show()