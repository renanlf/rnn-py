'''
Created on 22/09/2015

@author: renan
'''

from model.Neuron import Neuron

import numpy
from rbf.OutputNeuron import OutputNeuron

class RBF(object):
    '''
    classdocs
    '''


    def __init__(self, len_output_layer):
        '''
        Constructor
        '''
        self.__rbf_neurons = []
        self.__output_neurons = []
        self.__len_output_layer = len_output_layer
        
    def get_rbf_neurons(self):
        return self.__rbf_neurons
    
    def get_rbf_for_label(self, label):
        neurons = []
        
        for rbf_neuron in self.__rbf_neurons:
            numpy.equal
            if sum(rbf_neuron.get_label() == label) == len(label):
                
                neurons.append(rbf_neuron)
                
        return neurons
    
    def get_rbf_for_not_label(self, label):
        neurons = []
        
        for rbf_neuron in self.__rbf_neurons:
            if not sum(rbf_neuron.get_label() == label) == len(label):
                
                neurons.append(rbf_neuron)
                
        return neurons
    
    def output(self, pattern):
        result = numpy.zeros(self.__len_output_layer)
        
        for i, neuron in enumerate(self.__output_neurons):
            result[i] = neuron.output(rbf=self, pattern=pattern)
            
        return result
            
    def add_neuron(self, label):
        self.__output_neurons.append(OutputNeuron(label=label))
        
    
def gaussian(x):
    return 1.0/numpy.exp((-x)**2)
            