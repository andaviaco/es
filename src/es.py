import numpy as np
import random as rand
import pprint as pp

class Agent(object):
    """docstring for Agent"""
    def __init__(self, initial_solution, initial_strategy, initial_fitness):
        super(Agent, self).__init__()
        self.solution = initial_solution
        self.fitness = initial_fitness
        self.strategy = initial_strategy

    def __repr__(self):
        return f'<Agent s:{self.solution} m:{self.strategy} f:{self.fitness}>'


class ES(object):
    """docstring for ES"""
    def __init__(
        self,
        npopulation,
        ngenerations,
        fn_eval,
        *,
        lb=[-5, -5],
        ub=[5, 5],
    ):
        super(ES, self).__init__()

        self.npopulation = npopulation
        self.ngenerations = ngenerations
        self.fn_eval = fn_eval
        self.fn_lb = np.array(lb)
        self.fn_ub = np.array(ub)

    def optimize(self):
        self.population = self.initialize_population(self.npopulation)
        pp.pprint(self.population)

        for ngen in range(self.ngenerations):
            parents = self.select_parents()


    def initialize_population(self, npopulation):
        return [self.create_agent() for i in range(npopulation)]

    def select_parents(self):
        parent_1_index = self.random_parent_excluding()
        parent_2_index = self.random_parent_excluding([parent_1_index])

        parent_1 = self.population[parent_1_index]
        parent_2 = self.population[parent_2_index]

        return (parent_1, parent_2)

    def fitness(self, solution):
        fitness = 1 / (1 + self.fn_eval(solution))

        return fitness

    def create_agent(self):
        solution = self.random_vector(self.fn_lb, self.fn_ub)
        strategy = self.random_vector([0, 0], (self.fn_ub - self.fn_lb) * 0.5)
        fitness = self.fitness(solution)
        agent = Agent(solution, strategy, fitness)

        return agent

    def random_parent_excluding(self, excluded_index=[]):
        available_indexes = set(range(self.npopulation))
        exclude_set = set(excluded_index)
        diff = available_indexes - exclude_set
        selected = rand.choice(list(diff))

        return selected

    def random_vector(self, lb, ub):
        r = [rand.random() for i in lb]
        solution = lb + (ub - lb) * r

        return np.around(solution, decimals=4)
