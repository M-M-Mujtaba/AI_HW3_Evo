import numpy as np
from skimage import io
import os

filename = os.path.join(skimage.data_dir, 'moon.png')
moon = io.imread(filename)