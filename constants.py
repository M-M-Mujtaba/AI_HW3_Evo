import numpy as np
from skimage import io
messi = io.imread('face.png')
summ = np.sum(np.ones(messi.shape)*255, dtype=int)
mutation_size = 50
Population_Size = 100  # in 10^i
crossover_selection = 0.2
mutation_selection = 0.1