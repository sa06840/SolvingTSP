import tsplib95
import math
import random
import random as rm
import numpy as np
import csv

import copy
from matplotlib import pyplot as plt
from operator import add
from selectionSchemes import SelectionSchemes



class TSP(SelectionSchemes):
    #Initializes variables
    # Reads the data file and stores coordinates for each country (dictionary)
    def __init__(self, filename, populationSize, mutationRate, offspringsNumber, generations, iterations) -> None:
        self.populationSize = populationSize
        self.euclideanDistance = dict()      
        self.population=[]                  #stores all solutions
        self.listOfCountries=[]             #stores names of countries
        self.fitness=[]                     #stores total distance of every solution
        self.bestFitness= []
        self.averageFitness =[]
        self.numOfGenerations = generations
        self.numOfIterations  = iterations
        self.maxDistance = 0 #changed
        self.mutationRate = mutationRate
        self.offspringsNumber = offspringsNumber

        super()
        
        
        self.bestsofar = 9999999999

        problem = tsplib95.load(filename)
        temp = problem.as_name_dict()
        countryCoordinates= temp["node_coords"]
        self.generateEuclideanDistance(countryCoordinates)

    # Generates distance for each country with respect to each country (dictionary)   
    def generateEuclideanDistance(self,countryCoordinates):

        for country1 in countryCoordinates:
            distances=[]
            for country2 in countryCoordinates:
                distances.append(math.dist(countryCoordinates[country1], countryCoordinates[country2]))
            self.euclideanDistance[country1]= distances
            self.maxDistance += round(max(distances), 2)  #changed
        self.initializePopulation(countryCoordinates)

        # print(self.maxDistance)

    #Initializing a Population of 30 with a random shuffle of Countries
    def initializePopulation(self,countryCoordinates):
     
        for country in countryCoordinates:
            self.listOfCountries.append(country)

        for i in range(30):
            solution = copy.deepcopy(self.listOfCountries)
            random.shuffle(solution)
            self.population.append([0,solution])
        
    #Calculating Total Distance of each solution 
    def calculateFitness(self):
        count=0
        for solution in self.population:
            totalDistance=0
            for i in range(len(solution[1])-1):
                listOfDistance=self.euclideanDistance[solution[1][i]]
                distance=listOfDistance[(solution[1][i+1])-1]
                totalDistance+=distance
            '''
            Adding Distance of going back to origin country
            '''
            listOfDistance=self.euclideanDistance[solution[1][i+1]]
            distance=listOfDistance[solution[1][0]-1]
            totalDistance+=distance
            self.population[count][0]=self.maxDistance-round(totalDistance,2) #changed
            count+=1
       
    def newFitness(self, offspring):
        totalDistance=0
        distance = 0

        for i in range(len(offspring[1])-1):
            listOfDistance=self.euclideanDistance[offspring[1][i]]
            distance=listOfDistance[(offspring[1][i+1])-1]
            totalDistance+=distance

        '''
        Adding Distance of going back to origin country
        '''
        listOfDistance=self.euclideanDistance[offspring[1][i+1]]
        distance=listOfDistance[offspring[1][0]-1]
        totalDistance+=distance
        offspring[0]=self.maxDistance-round(totalDistance,2)   #changed

        # self.population.append(offspring)
        return (offspring)
    
    
    def crossover(self, parent1, parent2):
        #computing two random points to select elements from parents
        # self.assign_probabilities()
        p1 = parent1[1]          
        p2 = parent2[1]  
        
        start_index = random.randint(0, int(len(self.listOfCountries)/2))
        end_index = random.randint(start_index+1, len(self.listOfCountries)-1)

        offspring1 = [-1 for i in range(len(p1))]
        offspring1[start_index:end_index+1] = p1[start_index:end_index+1]

        counter = end_index+1
        for city in p2[end_index+1:]:
            if city not in offspring1:
                offspring1[counter] = city
                counter += 1

        if counter == len(p1):
            counter=0

        for city in p2[:end_index+1]:
            if city not in offspring1:
                offspring1[counter] = city
                counter+=1
            if counter == len(p1):
                counter=0
            if counter == start_index:
                break
        

        offspring2 = [-1 for i in range(len(p1))]
        offspring2[start_index:end_index+1] = p2[start_index:end_index+1]

        counter = end_index+1
        for city in p1[end_index+1:]:
            if city not in offspring2:
                offspring2[counter] = city
                counter += 1

        if counter == len(p2):
            counter=0

        for city in p1[:end_index+1]:
            if city not in offspring2:
                offspring2[counter] = city
                counter+=1
            if counter == len(p2):
                counter=0
            if counter == start_index:
                break

        offspring1 = [0, offspring1]
        offspring2 = [0, offspring2]
        
        return [offspring1, offspring2]

    def mutation(self, offspring):

        randomIndex1 = random.randint(0,len(self.listOfCountries)-1)
        randomIndex2 = random.randint(0,len(self.listOfCountries)-1)
        while randomIndex1 == randomIndex2:
            randomIndex2 = random.randint(0,len(self.listOfCountries)-1)
        
        country1 = offspring[1][randomIndex1]
        country2 = offspring[1][randomIndex2]
        offspring[1][randomIndex1] = country2
        offspring[1][randomIndex2] = country1

        return offspring
    

    def generationEvaluation(self):
        totalDistance = 0
        
        for chromosome in self.population:
            totalDistance += self.maxDistance-chromosome[0]
            if (self.maxDistance-chromosome[0]) < self.bestsofar:
                self.bestsofar = self.maxDistance-chromosome[0]

        self.averageFitness.append(totalDistance/len(self.population))
        self.bestFitness.append(self.bestsofar)


    def iterationEvaluation(self, fitnessEvaluation,iteration):
        # print(iteration)
        if iteration not in fitnessEvaluation:
            fitnessEvaluation[iteration] = [[],[]]
            fitnessEvaluation[iteration][0] = self.averageFitness 
            fitnessEvaluation[iteration][1] = self.bestFitness

          
    def plotGraphs(self, fitnessEvaluation):
        x_axis_generations = []
        addedAverageFitness = [0]*self.numOfGenerations
        addedBestFitness = [0]*self.numOfGenerations
        avgAverageFitness = []
        avgBestFitness = []
        data=[]
        data2=[]
        
        # heading_data = ["Run #1 BSF","Run #2 BSF","Run #3 BSF","Run #4 BSF","Run #5 BSF","Run #6 BSF","Run #7 BSF","Run #8 BSF","Run #9 BSF","Run #10 BSF" ]
        # data.append(heading_data)
        # list representing x axis (num of Generations)
        for i in range (1, self.numOfGenerations+1):
            x_axis_generations.append(i)
       
       #adding avgaveragefitness and best fitness values across all iterations
        count=1
        for iteration in (fitnessEvaluation):
            # print(row)
            # # 
            row1 = fitnessEvaluation[iteration][0]
            row1.insert(0, "Run #" + str(count) + " Average")
            data.append(row1)

            row2 = fitnessEvaluation[iteration][1]
            row2.insert(0, "Run #" + str(count) + " BSF")
            data2.append(row2)
        
            # print(fitnessEvaluation[iteration][0])
            addedAverageFitness = list(map(add, fitnessEvaluation[iteration][0][1:], addedAverageFitness))
            addedBestFitness  = list(map(add, fitnessEvaluation[iteration][1][1:], addedBestFitness ))
            count +=1
        #adjusting added avgavergaefitness and best fitness values
        #creating list representing y_axis (average avergae fitness values) and (average average best fitness values)
        for fitness in (addedAverageFitness):
            avgAverageFitness.append(fitness/self.numOfIterations)
        
        for fitness in (addedBestFitness):
            avgBestFitness.append(fitness/self.numOfIterations)


        data = np.array(data).T.tolist()
        data2 = np.array(data2).T.tolist()

        with open('T_AVG_Random_Random.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        with open('T_BEST_Random_Random.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data2)

        # data = {"Generation Index":x_axis_generations, "Average Average fitness ":avgAverageFitness, "Average Best Fitness":avgBestFitness}
        # df = pd.DataFrame(data)
        # df.to_csv("Generation_evaluation.csv", index=False)
       
        plt.plot(x_axis_generations, avgBestFitness, label = "Best Fitness")       
        plt.plot(x_axis_generations, avgAverageFitness,linestyle = "dashed", label = "Average Fitness")
        plt.xlabel("Number of Generations")
        plt.ylabel("Distance")
        plt.title("Travelling Salesman Problem (TSP)")
        plt.legend()
        plt.show()


 
