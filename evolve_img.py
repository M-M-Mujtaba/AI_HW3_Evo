
import numpy as np
import os
import skimage
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import random
from threading import Thread
from constants import messi
from constants import mutation_size
from constants import Population_Size
from constants import crossover_selection
from constants import mutation_selection
from constants import summ



class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def show_img(img):
    # print(np.sum(np.sum(messi, axis=0), axis=0))
    plt.imshow(img)
    plt.show()


# entity class which holds single image and its functional value
class Entity:

    def __init__(self):
        self.img = np.random.random_integers(0, 255, messi.shape)
        self.val = np.sum(np.abs(np.subtract(self.img, messi)))

    def update_val(self):
        self.val = np.sum(np.abs(np.subtract(self.img, messi)))

    @staticmethod
    def generate_pop(size):
        for i in range(size):
            yield Entity()


def crossover(plane1, plane2):
    x_point = random.randint(1, 249)
    y_point = random.randint(1, 249)
    # temp = plane1[:x_point, :y_point]
    # plane1[:x_point, :y_point] = plane2[:x_point, :y_point]
    # plane2[:x_point, :y_point] = temp
    #print(messi.shape[0])
    #print(messi.shape[1])
    temp = np.zeros((messi.shape[0], messi.shape[1]))
    temp[:x_point, :y_point] = plane1[:x_point, :y_point]
    temp[x_point:, :] = plane2[x_point:, :]
    temp[:, y_point:] = plane2[:, y_point:]

    return temp


def mutation(entity):
    positions = np.random.randint(0, 249 - mutation_size, (2, 4))
    # print(messi.shape[2])
    for i in range(messi.shape[2]):
        entity.img[positions[0][i]: positions[0][i] + mutation_size, positions[1][i]:positions[1][i] + mutation_size,
        i] = np.random.randint(0, 255, (mutation_size, mutation_size))


def Evolve():
    evolv_limit = 100
    evolv_index = 0
    Population = Entity.generate_pop(Population_Size)
    Population = sorted(Population, key=lambda e: e.val)
    best_entity = Population[0]
    new_poppulation = None
    show_img(best_entity.img)
    choices = [summ - Population[i].val  for i in range(Population_Size)]
    while best_entity.val != 0 and evolv_index < evolv_limit:
        x = Population[0]
        y = Population[1]
        new_poppulation = []

        for i in range(Population_Size):
            child = Entity()
            threadss = []
            for j in range(messi.shape[2]):
                threeadd = ThreadWithReturnValue(target=crossover, args=(x.img[:, :, j], y.img[:, :, j]))
                threadss.append(threeadd)
            for j in range(messi.shape[2]):
                threadss[j].start()
            for j in range(messi.shape[2]):
                child.img[:, :, j] = threadss[j].join()
            if random.randint(1, 10) == 10:
                mutation(child)
            child.update_val()
            if child.val > best_entity.val:
                best_entity = child
                show_img(best_entity.img)
            new_poppulation.append(child)
        evolv_index += 1
        print(evolv_index)
        Population = sorted(new_poppulation, key= lambda e:e.val)
        choices = [summ - Population[k].val  for k in range(Population_Size)]


# threads = []
#
# for i in range(messi.shape[2]):
#     threead = ThreadWithReturnValue(target=crossover, args=(messi[:, :, i], messi[:, :, i]))
#     threads.append(threead)
# for i in range(messi.shape[2]):
#     threads[i].start()
# for i in range(messi.shape[2]):
#     messi[:, :, i] = threads[i].join()
# messi_Entity = Entity()
# mutation(messi_Entity)

Evolve()

