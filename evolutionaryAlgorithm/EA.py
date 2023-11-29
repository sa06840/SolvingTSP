import random
from tsp import TSP


def evolutionaryAlgorithm(filename, populationSize, mutationRate, offspringsNumber, generations, iterations):

    for iteration in range(iterations):

        print("***** Iteration Number = " + str(iteration+1) + " *****")

        tsp = TSP(filename, populationSize, mutationRate, offspringsNumber, generations, iterations)
        tsp.calculateFitness()

        for generation in range(tsp.numOfGenerations):

            totalOffsprings = []

            for i in range(tsp.offspringsNumber//2):
                parents = tsp.randomSelection(0)
                # parents = tsp.fpsSelection(0)
                # parents = tsp.rbsSelection(0)
                # parents = tsp.truncation(0)
                # parents = tsp.binarySelection(0)

                p1 = parents[0]
                p2 = parents[1]

                offsprings = tsp.crossover(p1, p2)

                for j in range(2):
                    randomNumber = round(random.uniform(0.00, 1.00), 2)
                    if randomNumber < tsp.mutationRate:

                        tempOffspring = tsp.mutation(offsprings[j])
                        offsprings[j] = tempOffspring

                    offspring = tsp.newFitness(offsprings[j])

                    totalOffsprings.append(offspring)
            
            for i in totalOffsprings:
                tsp.population.append(i)

            # tsp.randomSelection(1)
            # tsp.fpsSelection(1)
            # tsp.rbsSelection(1)
            tsp.truncation(1)
            # tsp.binarySelection(1)


            # tsp.generationEvaluation()

        # tsp.iterationEvaluation(fitnessEvaluation,iteration)
        tsp.population.sort()
        tsp.population.reverse()
        print(tsp.maxDistance - tsp.population[0][0])
        
    # tsp.plotGraphs(fitnessEvaluation)



tspFile ='/Users/sajeelnadeemalam/Documents/ArtificialIntelligence/AI-Project/evolutionaryAlgorithm/qa194.tsp'

evolutionaryAlgorithm(tspFile, 30, 0.2, 10, 10000, 10)