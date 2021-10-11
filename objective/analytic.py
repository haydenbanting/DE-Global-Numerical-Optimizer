import numpy as np

def ackley2D(x, y):
    

    f = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2) )) - np.exp( 0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)) )  + np.e + 20  

    return f

def ackley(x, n=2):

    # Evaluate the sums 
    s1, s2 = 0, 0
    for i in range(n):
        s1 += x[i]**2
        s2 += np.cos(2*np.pi*x[i])

    return -20 * np.exp( -0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e


