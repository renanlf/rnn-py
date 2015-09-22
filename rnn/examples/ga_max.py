'''
Created on 15/09/2015

@author: renan
'''
from ga.Genetic_Algorithm import Genetic_Algorithm

import numpy


def fenotype(individual):
    return numpy.sum(individual.get_genotype())

if __name__ == '__main__':
    ga = Genetic_Algorithm(individuals=4,
                           mutation_rate=0.1,
                           genotype=5,
                           fenotype=fenotype,
                           genotype_type=Genetic_Algorithm.TYPE_BINARY)
    
    ga.print_individuals()
    ga.execute(generations=10)
    ga.print_individuals()
    print "Best Individual for all generations", ga.get_best_individual().get_genotype(), ga.get_best_individual_fenotype()
