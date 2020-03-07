import numpy as np
import matplotlib.pyplot as plt
import random
from threading import Thread
from skimage import io
from constants import mutation_size
from constants import Population_Size
from constants import crossover_selection
from playsound import playsound



class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
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

# needed to create and object to send to the threads
class Frame:
    def __init__(self, img):
        self.img = img


# entity class which holds single image and its functional value
class Entity:

    def __init__(self, target):  # target is the 2d image array that we have to get to
        self.target = target.img
        self.img = np.random.random_integers(0, 255,
                                             self.target.shape)  # create a random image of the shape of target image
        self.val = 1 - (np.sum(np.abs(np.subtract(self.img, self.target))) / (
                self.target.size * 255))  # divide it by the max distance the find how close we are

    def update_val(self):  # update the evaluation value when the image is modified with crossover or mutation
        self.val = 1 - (np.sum(np.abs(np.subtract(self.img, self.target))) / (
                self.target.size * 255))  # divide it by the max distance the find how close we are

    @staticmethod
    def generate_pop(size, target):  # return a list of randomly generated population
        for i in range(size):
            yield Entity(target)


# Return a new child created by combining two planes with random breaking point
def crossover(plane1, plane2):
    x_point = random.randint(1, len(plane1) - 1)
    y_point = random.randint(1, len(plane1) - 1)
    child = np.zeros((len(plane1),len(plane1)))  # initialize the child to 0s
    child[:x_point, :y_point] = plane1[:x_point, :y_point]  # copy plane 1 from initial to xpoint and ypoint in child
    child[x_point:, :] = plane2[x_point:, :]  # copy plane 2 from xpoint and ypoint till the end
    child[:, y_point:] = plane2[:, y_point:]

    return child


# apply grid level mutaion(randomly change value)
def mutation(entity):
    positions = np.random.randint(0, len(entity.img) - mutation_size - 1, (2, 1))  # generating a starting point
    # between 0 and max size - mutation size
    # the image is 2d numpy array
    entity.img[positions[0][0]: positions[0][0] + mutation_size,
    positions[1][0]:positions[1][0] + mutation_size] = np.random.randint(0, 255, (mutation_size, mutation_size))


# the main evolutionary function that apply crossover and mutation to randomly generated images in order to converge to
# the target image
def Evolve(plane):
    evolv_limit = 1000  # maximum number of generations
    evolv_index = 0  # index for generations
    Population = Entity.generate_pop(Population_Size, plane)
    Population = sorted(Population, key=lambda e: e.val, reverse=True)  # sort the populations based on the fitness
    # value , maximum values at top

    best_entity = Population[0]  # the first image is the best randomly generated image
    # show_img(best_entity.img)
    pop_square = Population_Size * Population_Size
    choices = [Population[i].val * (pop_square - i * i) for i in range(Population_Size)]  # The Roulet
    # wheel to decide which child to be selected, probability of selection would be proportional to its fitness value

    # run until we converge to the target or we reach our generational limit
    while best_entity.val != 1 and evolv_index < evolv_limit:
        x = random.choices(Population, weights=choices, k=int(Population_Size * crossover_selection))  # get the top
        # crossover_(selection * 100) % of populations based on their proportional probability
        y = random.choices(Population, weights=choices, k=int(Population_Size * crossover_selection))
        new_population = []
        mutation_selection = (((evolv_limit + 1 - evolv_index) / evolv_limit) * 0.5 + 0.05)
        for i in range(Population_Size):
            parent1 = x[i % int(Population_Size * crossover_selection)]
            parent2 = y[i % int(Population_Size * crossover_selection)]
            child = Entity(plane)
            child.img = crossover(parent1.img, parent2.img)
            if random.randint(0, int(1 / mutation_selection)) == 1:
                mutation(child)
                # print("mutated")
            child.update_val()
            new_population.append(child)

        evolv_index += 1
        print(evolv_index)
        Population = sorted(new_population, key=lambda e: e.val, reverse=True)
        if Population[0].val > best_entity.val:
            print("New best found at generation {} with val{}".format(evolv_index, best_entity.val))
            best_entity = Population[0]
            # show_img(best_entity.img)
        choices = [Population[i].val * (pop_square - i * i) for i in range(Population_Size)]

    return best_entity.img
    # threads = []



def main():
    messi = io.imread('low_res.jpg')
    print(messi.shape)
    final_img = np.zeros(messi.shape, dtype=int)
    Threads = []
    Music = Thread(target=playsound, args=('boomboom.mp3',) )
    Music.start()
    # create thread for each layer and evolve them simultaneously
    for i in range(messi.shape[2]):
        arg_to_send = Frame(messi[:, :, i])
        Threadd = ThreadWithReturnValue(target=Evolve, args=(arg_to_send,))
        Threads.append(Threadd)
    for i in range(messi.shape[2]):
        Threads[i].start()
    for i in range(messi.shape[2]):
        final_img[:, :, i] = Threads[i].join()
    Music.join()
    Music.start()
    Music.join()
    # for i in range(messi.shape[2]):
    #     final_img[:,:,i] = messi[:,:,i]
    show_img(final_img)

if __name__ == "__main__":
    main()
