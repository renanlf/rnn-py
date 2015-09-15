'''
Created on 18/08/2015

@author: renan
'''
import numpy
from Model.Neuron import Neuron
class Layer(object):
    '''
    classdocs
    '''
    lastLayer = False
    neurons = []

    def __init__(self, lastLayer = False):
        '''
        Constructor
        '''
        self.lastLayer = lastLayer
        self.neurons = []
    
    def addNeuron(self, neuron):
        self.neurons.append(neuron)
        
    def output(self, inputs):
        result = numpy.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            out = neuron.output(inputs)
            result[i] = out
        return result
    
    def updateDelta(self, inputs, error):
        for neuron in self.neurons:
            neuron.updateDelta(inputs, error)
            
    def update(self):
        for neuron in self.neurons:
            neuron.update()
    