# encoding: utf-8
'''
Created on 22/09/2015

@author: renan
'''

import numpy

class RBFNeuron(object):
    '''
    Esta classe representa um neurônio da camada RBF de uma rede neural
    '''


    def __init__(self, centroid, label, radius=0):
        '''
        Constructor
        '''
        self.__radius = radius
        self.__centroid = centroid
        self.__weight = 1.0
        self.__label  = label
        
    def output(self, t):
        distance = numpy.sum(self.__centroid - t)**2
        
        return numpy.exp(- distance / self.__radius**2)
        
    def get_radius(self):
        return self.__radius
    
    def set_radius(self, radius):
        self.__radius = radius
        
    def get_centroid(self):
        return self.__centroid
    
    def get_weight(self):
        return self.__weight
    
    def set_weight(self, weight):
        self.__weight = weight
        
    def get_label(self):
        return self.__label