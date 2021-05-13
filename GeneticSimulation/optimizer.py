import random
from numpy.random import permutation

from model_functions import createRandIndividual, fitness_score


def FitnessScore(individual):
    name = individual["name"]
    print(f"{name} is living")
    score = fitness_score(individual)
    individual["health_score"] = score
    print(f"{name} achieved a score of {score}")
    return individual


class Optimizer:
    choices = {}

    def __init__(self, choices={}):
        self.mutateProb = 0.2
        self.randomSelect = 0.25
        self.retain = 0.3
        for k, v in choices:
            self.choices[k] = v

    def getFitnessScore(self, individual):
        score = fitness_score(individual)
        individual["health_score"] = score
        return individual

    def createPopulation(self, numIndividuals: int):
        population = []
        for _ in range(numIndividuals):
            individual = createRandIndividual()
            population.append(individual)
        return population

    def breed(self, mother: dict, father: dict):
        parents_genes = {k: [mother[k], father[k]] for k in mother.keys()}

        children = []

        for _ in range(2):
            child = {}
            for gene in parents_genes:
                if gene == "layers":
                    choice = list(permutation([0, 1]))
                    c_layers = []
                    for i, layer in enumerate(parents_genes[gene][choice[0]]):
                        p_layers = [layer]
                        if i <= len(parents_genes[gene][choice[1]]) - 1:
                            p_layers.append(parents_genes[gene][choice[1]][i])
                        c_layers.append(random.choice(p_layers))
                    child[gene] = c_layers
                else:
                    child[gene] = random.choice(parents_genes[gene])

            if self.mutateProb > random.random():
                child = self.mutate(child)

            children.append(child)

        return children

    def mutate(self, child):
        rand = createRandIndividual()
        gene = random.choice(list(rand.keys()))
        child[gene] = rand[gene]
        return child

    def getAvgScore(self, population):
        return float(
            sum([individual["health_score"]
                 for individual in population]) / len(population))

    def evolve(self, population):
        sortedPopulation = sorted(population,
                                  key=lambda k: k['health_score'],
                                  reverse=True)
        print(f"Population size is {len(population)}")

        retainLength = int(len(sortedPopulation) * self.retain)

        parents = sortedPopulation[0:retainLength]

        print(
            f"Population size after selecting healthy parents is {len(parents)}"
        )

        for individual in sortedPopulation[retainLength:]:
            if self.randomSelect > random.random():
                parents.append(individual)

        print(
            f"Population size after a random selection of unhealthy parents is {len(parents)}"
        )

        desiredChildren = len(population) - len(parents)

        print(f"Desired children {desiredChildren}")

        children = []

        while len(children) < desiredChildren:
            male = random.choice(parents)
            female = random.choice(parents)

            if male != female:
                babies = self.breed(male, female)
                children.extend(babies)

        parents.extend(children)

        print(f"New Generation size: {len(parents)}")

        return parents