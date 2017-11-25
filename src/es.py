import numpy as np
import random as rand
import pprint as pp
from operator import itemgetter, attrgetter

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
        # pp.pprint(self.population)

        for ngen in range(self.ngenerations):
            parents = self.select_parents()
            child = self.recombination_discrete(*parents)

            new_population = [*self.population, child]

            child.solution = self.covariance_vector(new_population, len(child.solution))
            child.fitness = self.fitness(child.solution)

            inferior_agent_index = self.inferior_agent_index(new_population)
            del new_population[inferior_agent_index]

            self.population = new_population

        # pp.pprint(self.population)
        best = self.get_best(self.population)

        return best.solution


    def initialize_population(self, npopulation):
        return [self.create_agent() for i in range(npopulation)]

    def select_parents(self):
        parent_1_index = self.random_parent_excluding()
        parent_2_index = self.random_parent_excluding([parent_1_index])

        parent_1 = self.population[parent_1_index]
        parent_2 = self.population[parent_2_index]

        return (parent_1, parent_2)

    def recombination_discrete(self, parent_1, parent_2):
        dim = len(parent_1.solution)

        solution = [parent_1.solution[i] if rand.randint(0, 1) else parent_2.solution[i] for i in range(dim)]
        strategy = [parent_1.strategy[i] if rand.randint(0, 1) else parent_2.strategy[i] for i in range(dim)]

        solution = self.round_vector(solution)
        strategy = self.round_vector(strategy)
        fitness = self.fitness(solution)

        child = Agent(solution, strategy, fitness)

        return child

    def recombination_intermediate(self, parent_1, parent_2):
        pass

    def covariance_vector(self, population, size):
        variances = np.vstack((agent.strategy for agent in population))
        covariance_matrix = np.cov(variances)
        diagonal_sum = np.sum(covariance_matrix.diagonal())
        rand_vector = np.random.normal(0, diagonal_sum, size)
        rand_vector = self.round_vector(rand_vector)

        return rand_vector

    def inferior_agent_index(self, population):
        fitness_enum = enumerate([agent.fitness for agent in population])
        index, _ = min(fitness_enum, key=itemgetter(1))

        return index

    def get_best(self, population):
        best = max(population, key=attrgetter('fitness'))

        return best

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

        return self.round_vector(solution)

    def round_vector(self, vector):
        return np.around(vector, decimals=4)
