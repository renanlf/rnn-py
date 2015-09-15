'''
Created on 15/09/2015

@author: renan
'''
import numpy

class Individual(object):
    '''
    classdocs
    '''


    def __init__(self, genotype):
        '''
        Constructor
        '''
        self.__genotype = genotype
        
    def get_genotype(self):
        return self.__genotype
    
    def set_genotype(self, genotype):
        self.__genotype = genotype