'''
Created on 18/08/2015

@author: renan
'''
import numpy

class Neuron(object):
    
    
#     warnings.filterwarnings('error')
    '''
    classdocs
    '''

    def __init__(self, weights, function, n = 0.05, bias = 0, momentum = 0.1):
        '''
        Constructor
        '''
        # caso queira passar os pesos ou deixar que o construtor os crie
        if(type(weights) == int):
            self._weights = numpy.float128(numpy.random.uniform(-0.5,0.5,weights))
            self.__deltas = numpy.zeros(weights)
        else:
            self._weights = weights
            self.__deltas = numpy.zeros(len(weights))
            
        self.__function = function
        self.__n = n
        self._bias = numpy.float128(numpy.random.uniform(-0.5,0.5,1))
        self.__deltaBias = 0
        self._sigma = 0
        
        self.__momentum = momentum
        
    def _output(self, inputs):
        self._inputs = inputs
        try:
            E = inputs * self._weights
        except RuntimeWarning:
            print inputs, self._weights
        
        self._a = numpy.sum(E) + self._bias
        result = self.__function(self._a)
        
        return result
        
    def _updateDelta(self):
        
        self.__deltas = self.__deltas +  self.__n * self._sigma * self._inputs
#         self.__deltas = self.__deltas + self.__momentum * self.__deltas
        
        self.__deltaBias = self.__deltaBias + self.__n * self._sigma
        
    def _update(self):
        self._weights = self._weights + self.__deltas       
            
        self._bias = self._bias + self.__deltaBias
        
        self.__deltas = numpy.zeros(len(self._weights))
        self.__deltaBias = 0
        
    def _getWeight(self, n):
        return self._weights[n]
    