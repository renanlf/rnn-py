'''
Created on 15/09/2015

@author: renan
'''
from ga.Individual import Individual
import numpy

class Genetic_Algorithm(object):
    '''
    classdocs
    '''
    TYPE_FLOAT = 1
    TYPE_INT   = 0
    TYPE_BINARY= 2
    
    def create_individuals(size, length_genotype, genotype_type):
        individuals = list()
        
        if(genotype_type == Genetic_Algorithm.TYPE_FLOAT):            
            for i in range(0, size):
                genotype = numpy.random.uniform(low = -0.5, high = 0.5, size = length_genotype)
                individuals[i] = Individual(genotype)
                
        elif(genotype_type == Genetic_Algorithm.TYPE_BINARY):
            for i in range(0, size):
                genotype = numpy.random.randint(low = 0, high = 2, size = length_genotype)
                individuals.append(Individual(genotype))
                
        return individuals   
            
    create_individuals = staticmethod(create_individuals)

    def __init__(self, individuals, mutation_rate, cross_rate, genotype, fenotype, genotype_type = TYPE_FLOAT):
        '''
        Constructor
        '''
        self.__mutation_rate = mutation_rate
        self.__cross_rate = 0.5
        self.__genotype_type = genotype_type
        self.__length_genotype = genotype
        self.__fenotype = fenotype
        
        self.__individuals = Genetic_Algorithm.create_individuals(individuals, genotype, genotype_type)
        
    def crossover(self, individual_a, individual_b):
            if(self.__genotype_type == Genetic_Algorithm.TYPE_FLOAT):
                return None
            
            elif(self.__genotype_type == Genetic_Algorithm.TYPE_BINARY):
                                  
                position = numpy.random.randint(low = 0, high = self.__length_genotype, size = 1)
                    
                new_genotype_a = numpy.zeros(self.__length_genotype)
                new_genotype_b = numpy.zeros(self.__length_genotype)
                    
                new_genotype_a[0:position] = individual_a.get_genotype()[0:position]
                new_genotype_b[0:position] = individual_b.get_genotype()[0:position]
                    
                new_genotype_a[position:self.__length_genotype] = individual_a.get_genotype()[position:self.__length_genotype]
                new_genotype_b[position:self.__length_genotype] = individual_b.get_genotype()[position:self.__length_genotype]
                    
                new_individual_a = Individual(new_genotype_a)
                new_individual_b = Individual(new_genotype_b)
                    
                return new_individual_a, new_individual_b
                
    def mutate(self, individual):
        if(self.__genotype_type == Genetic_Algorithm.TYPE_FLOAT):
            return None
        
        elif(self.__genotype_type == Genetic_Algorithm.TYPE_BINARY):
            probs = numpy.random.random(self.__length_genotype)
            mut   = probs < self.__mutation_rate
            
            for i in range(0, self.__length_genotype):
                if(mut[i]):
                    individual.get_genotype()[i] = not individual.get_genotype()[i]
                    
    def execute(self, generations = 1):
        fenotypes = numpy.zeros(len(self.__individuals))
        
        for generation in (0, generations):
            
            new_generation = list()
            
            for i, individual in enumerate(self.__individuals):
                fenotypes[i] = self.__fenotype(individual)
            
            best_individual = numpy.argmax(fenotypes)
            self.__best_individual = self.__individuals[best_individual]
            
            #normalizando os fenotipos
            fenotypes = fenotypes/numpy.sum(fenotypes)
            
            for k in range(0, int(len(self.__individuals)/2)):         
            
                probs = numpy.random.random(2)
                
                # select two individuals
                selected_individuals = list()
                i = 0
                value = 0
                top_value = 0
                while(len(selected_individuals) < 2):
                    top_value += fenotypes[i]
                    if(probs[0] < top_value and probs[0] >= value):
                        selected_individuals.append(self.__individuals[i])
                        
                    if(probs[1] < top_value and probs[1] >= value):
                        selected_individuals.append(self.__individuals[i])
                        
                    i += 1
                    value = top_value
                    
                new_individual_a, new_individual_b = self.crossover(selected_individuals[0], selected_individuals[1])
                
                self.mutate(new_individual_a)
                self.mutate(new_individual_b)
                
                new_generation.append(new_individual_a)
                new_generation.append(new_individual_b)
                
            self.__individuals = new_generation
            
    def get_best_individual(self):
        return self.__best_individual
            
    def print_individuals(self):
        for individual in self.__individuals:
            print individual.get_genotype()
        