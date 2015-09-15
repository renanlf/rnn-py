'''
Created on 15/09/2015

@author: renan
'''

from ga.Genetic_Algorithm import Genetic_Algorithm

import numpy

def get_data():
    train, target = numpy.loadtxt(fname = '../../../data', 
                                  usecols = (0,1,2,3),
                                  delimiter = ',')
    
    print train, target
def fenotype(individual):
    genotype = individual.get_genotype()

if __name__ == '__main__':
    get_data()
    ga = Genetic_Algorithm(individuals = 4, 
                           mutation_rate = 0.1,
                           genotype = 4, 
                           fenotype = fenotype, 
                           genotype_type = Genetic_Algorithm.TYPE_FLOAT)
    
    ga.print_individuals()
    ga.execute(generations = 10)
    ga.print_individuals()
    print "Best Individual for last generation", ga.get_best_individual().get_genotype()