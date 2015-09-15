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
        self.__lastLayer = lastLayer
        self._neurons = []
    
    def _addNeuron(self, neuron):
        self._neurons.append(neuron)
        
    def _output(self, inputs):
        result = numpy.zeros(len(self._neurons))
        for i, neuron in enumerate(self._neurons):
            out = neuron._output(inputs)
            result[i] = out
        return result
    
    def _updateDelta(self, inputs, error):
        for neuron in self._neurons:
            neuron._updateDelta(inputs, error)
            
    def _update(self):
        for neuron in self._neurons:
            neuron._update()
    