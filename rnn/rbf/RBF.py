'''
Created on 22/09/2015

@author: renan
'''

from model.Neuron import Neuron

import numpy

class RBF(object):
    '''
    classdocs
    '''


    def __init__(self, len_output_layer):
        '''
        Constructor
        '''
        self.__rbfneurons = []
        self.__output_neurons = []
        self.__len_output_layer = len_output_layer
            
def gaussian(x):
    return 1.0/numpy.exp((-x)**2)
            