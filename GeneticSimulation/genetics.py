import os

from numpy import isnan
from multiprocessing import pool, cpu_count
from optimizer import Optimizer, FitnessScore
from options import MAX_PROCESSES, POPULATION_STORAGE
from json import dumps
from datetime import datetime


def diedoff(population):
    livingpopulation = []
    for individual in population:
        if not isnan(individual["health_score"]):
            livingpopulation.append(individual)
    return livingpopulation


def start_life(generations: int, population: int):
    life = Optimizer()

    if not os.path.isdir(POPULATION_STORAGE):
        print('The population directory is not present. Creating a new one..')
        os.mkdir(POPULATION_STORAGE)

    population = life.createPopulation(population)

    for i in range(generations):
        for identifier in range(len(population)):
            population[identifier]["name"] = identifier

        print(f"Generation {i} is born\n")

        start_time = datetime.now()

        with open(f"./{POPULATION_STORAGE}/generation{i}", 'w') as f:
            f.write(f"# Generation {i} is born\n")

        with pool.Pool(min(cpu_count(), MAX_PROCESSES)) as p:
            population = p.map(FitnessScore, population)

        # Did anyone die during their life?
        population = diedoff(population)

        print("[living complete]")

        end_time = str(datetime.now() - start_time)

        averageScore = life.getAvgScore(population)

        with open(f"./{POPULATION_STORAGE}/generation{i}", 'a') as f:
            f.write(f"# Time to live: {end_time}\n")
            f.write(f"# Average Health Score {averageScore}\n")
            f.write(
                dumps(sorted(population,
                             key=lambda k: k['health_score'],
                             reverse=True),
                      indent=2))

        if i < generations:
            population = life.evolve(population)

        print("[evolution complete]")


if __name__ == '__main__':
    start_life(20, 50)
