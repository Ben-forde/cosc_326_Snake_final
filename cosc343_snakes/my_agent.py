__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import random
from math import e
import numpy as np

agentName = "<my_agent>"
perceptFieldOfVision = 5  # Choose either 3,5,7 or 9
perceptFrames = 1  # Choose either 1,2,3 or 4
trainingSchedule = [("self", 1), ("random", 499)]
average_array = list()


# This is the class for your snake/agent


class Snake:

    def __init__(self, nPercepts, actions):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values)

        self.nPercepts = nPercepts
        self.actions = actions
        self.chromosomes = np.random.randint(low=-25, high=25, size=(perceptFieldOfVision * perceptFieldOfVision) * 3)
        self.bias_left = random.randint(-25,25)
        self.bias_foward = random.randint(-25,25)
        self.bias_right = random.randint(-25,25)


    def AgentFunction(self, percepts):

        index = self.evaluate_perceptron(percepts)
        return self.actions[index]

    def perceptron_left(self, percepts):
        """
        perceptron for which if it has the highest evaluation will give the action turning left.
        :param percepts: given information on snakes surroundings.
        :return: the sum of inputs*chromosome_weights + bias
        """
        training_inputs = percepts.flatten()
        end = int(len(self.chromosomes)/3)
        chrome_segment = self.chromosomes[0:end]
        output_sum = np.dot(training_inputs, chrome_segment)
        output_sum += self.bias_left
        return output_sum

    def perceptron_foward(self, percepts):
        """
        perceptron for which if it has the highest evaluation will give the action of going foward
        :param percepts: given information on snakes surroundings.
        :return: the sum of inputs*chromosome_weights + bias
        """
        training_inputs = percepts.flatten()
        start = int(len(self.chromosomes)/3)
        end = (start *2)
        chrome_segment = self.chromosomes[start:end]
        output_sum = np.dot(chrome_segment, training_inputs)
        output_sum += self.bias_foward
        return output_sum

    def perceptron_right(self, percepts):
        """
        perceptron for which if it has the highest evaluation will give the action turning right.
        :param percepts: given information on snakes surroundings.
        :return: the sum of inputs*chromosome_weights + bias
        """
        training_inputs = percepts.flatten()
        start = (int((len(self.chromosomes)/3) * 2))
        end = len(self.chromosomes)
        chrome_segment = self.chromosomes[start:end]
        output_sum = np.dot(training_inputs, chrome_segment)
        output_sum += self.bias_right
        return output_sum

    def evaluate_perceptron(self, percepts):
        """
        gets the move from the model
        :param percepts: given information on snakes surroundings.
        :return: the index of what action to take.
        """
        foward_sum = self.perceptron_foward(percepts)
        left_sum = self.perceptron_left(percepts)
        right_sum = self.perceptron_right(percepts)
        list_of_sum = [left_sum, foward_sum, right_sum]
        list_of_sum.sort()
        if list_of_sum[0] == left_sum:
            return 0
        elif list_of_sum[0] == foward_sum:
            return 1
        else:
            return 2


def evalFitness(population):
    """
    alterted this function to calculate average size and turns per snake,
    which is used evaluate the snakes differently then the given function.
    the new evaluation gives a higher score for those snakes which are longer
    and stay alive for more turns
    :param population: the current population of snakes.
    :return: fitness which array of fitnesses.
    """
    N = len(population)
    fitness = np.zeros((N))
    sum_size = 0
    total_turns_alive = 0

    for g, snake in enumerate(population):
        sum_size += np.max(snake.sizes)
        total_turns_alive += np.sum(snake.sizes > 0)

    avg_size = sum_size / len(population)
    avg_turns = total_turns_alive / 100

    for n, snake in enumerate(population):

        maxSize = np.max(snake.sizes)
        turnsAlive = np.sum(snake.sizes > 0)
        maxTurns = len(snake.sizes)
        if maxSize > avg_size:
            if maxSize / avg_size > 2:
                maxSize = maxSize * 2
            elif 1.5 < maxSize / avg_size < 2:
                maxSize = maxSize * 1.7
            else:
                maxSize = maxSize * 1.3

        if turnsAlive > avg_turns:
            turnsAlive = turnsAlive * 1.2


        fitness[n] = maxSize + turnsAlive / maxTurns

        average_array.append(maxSize + turnsAlive / maxTurns)

    return fitness


def create_percent_map(snake_tuple):
    """
    creates a percent map which is needed for the selection of pearents.
    where i have decided to use roulette wheel selection.
    :param snake_tuple: takes a tuple which is [snake_object, fitness_score]
    :return: a list of percentatges
    """
    sum_fit = 0
    for i in snake_tuple:
        sum_fit += i[1]
    p = []
    for i in snake_tuple:
        p.append(i[1] / sum_fit)
    return p


def select_chromosome_per_neuron(pearent1, pearent2):
    """
    this function does creates the crossover for each child by going through in thirds
    for each neuron (I split my chromosomes in three, one third for each percpetron)
    then once through all loops calls chrome_mutation which injects mutation into the child.
    :param pearent1: snake object
    :param pearent2: snake object
    :return: new chromosomes
    """
    mutation_rate = 2400
    counter = perceptFieldOfVision
    new_child_chromosomes = []
    flag = True


    for i in range(0, int(len(pearent1[0].chromosomes))):
        mutation = random.randint(0, mutation_rate)
        if mutation == 1:
            new_child_chromosomes.append(random.randint(-25,25))
        else:
            if flag == True:
                new_child_chromosomes.append(pearent1[0].chromosomes[i])
                counter -= 1
                if counter == 0:
                    flag = False
                    counter = perceptFieldOfVision
            else:
                new_child_chromosomes.append(pearent2[0].chromosomes[i])
                counter -= 1
                if counter == 0:
                    flag = True
                    counter = perceptFieldOfVision

    return chrome_mutation(new_child_chromosomes)

def chrome_mutation(new_child_chromosomes):
    """
    injects mutation into the new child.
    :param new_child_chromosomes: a set of chromosomes.
    :return: a new set of chromosomes potentionlly with mutation injected.
    """

    for i in range(0, len(new_child_chromosomes)-1):
        mut = random.randint(0, (perceptFieldOfVision ** 3) * 7)
        if mut == 1:
            new_child_chromosomes[i] = random.randint(-25,25)

    return new_child_chromosomes




def new_bias(pearent1, pearent2):
    """
    selects which bias from each pearent to give to new snake.
    also inject mutation into the biases
    :param pearent1: snake_object
    :param pearent2: snake_object
    :return: the new bias with crossover and mutation
    """

    new_left_bias = pearent1.bias_left
    new_foward_bias = pearent1.bias_foward
    new_right_bias = pearent1.bias_right

    left_mutation = random.randint(0, 1)
    foward_mutation = random.randint(0, 1)
    right_mutation = random.randint(0, 1)

    new_change = random.randint(0, 100)
    if new_change == 1:

        new_left_bias = random.randint(-25,25)
        new_foward_bias = random.randint(-25,25)
        new_right_bias = random.randint(-25,25)
        return [new_left_bias, new_foward_bias, new_right_bias]
    else:
        if left_mutation == 1:
            new_left_bias = pearent2.bias_left
        if foward_mutation == 1:
            new_foward_bias = pearent2.bias_foward
        if right_mutation == 1:
            new_right_bias = pearent2.bias_right
        return [new_left_bias, new_foward_bias, new_right_bias]


def newGeneration(old_population):
    """
    :param old_population: previous snakes used for generating
    the new generation.
    :return: tuple of consisting of a list of new snakes and the average fittnes
    """
    N = len(old_population)
    nPercepts = old_population[0].nPercepts
    actions = old_population[0].actions
    snake_tuple = dict()
    fitness = evalFitness(old_population)
    for i in range(0, len(fitness) - 1):
        snake_tuple[old_population[i]] = fitness[i]

    snake_tuple = sorted(snake_tuple.items(), key=lambda x: x[1], reverse=True)
    p = create_percent_map(snake_tuple)
    new_population = list()

    for n in range(N):
        pearent1, pearent2 = np.random.choice(len(snake_tuple), 2, False, p)
        new_snake = Snake(nPercepts, actions)
        child_snake_chromosomes = select_chromosome_per_neuron(snake_tuple[pearent1],
                                                               snake_tuple[pearent2])  # _per_neuron
        new_snake.chromosomes = child_snake_chromosomes
        bias = new_bias(snake_tuple[pearent1][0], snake_tuple[pearent2][0])
        new_snake.bias_left = bias[0]
        new_snake.bias_foward = bias[1]
        new_snake.bias_right = bias[2]
        new_population.append(new_snake)

        avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
