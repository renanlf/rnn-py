'''
Created on 15/09/2015

@author: renan
'''
from ga.Genetic_Algorithm import Genetic_Algorithm
from model.MLP import MLP

def fenotype(genotype):
    return sum(genotype)


if __name__ == '__main__':
    