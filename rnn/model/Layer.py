'''
Created on 18/08/2015

@author: renan
'''
import numpy

class Layer(object):
    '''
    classdocs
    '''

    def __init__(self, lastLayer = False):
        '''
        Constructor
        '''
        self.__neurons = []
    
    def add_neuron(self, neuron):
        self.__neurons.append(neuron)
        
    def output(self, inputs):
        result = numpy.zeros(len(self.__neurons))
        for i, neuron in enumerate(self.__neurons):
            output = neuron.output(inputs)
            #optimizing
            result.itemset(i, output)
        return result
            
    def get_neurons(self):
        return self.__neurons
    