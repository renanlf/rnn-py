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
            self.__weights = numpy.float128(numpy.random.uniform(-0.5,0.5,weights))
            self.__Deltas = numpy.zeros(weights)
        else:
            self.__weights = weights
            self.__Deltas = numpy.zeros(len(weights))
            
        self.__function = function
        self.__n = n
        self.__bias = numpy.float128(numpy.random.uniform(-0.5,0.5,1))
        self.__Delta_bias = 0
        self.__delta = 0
        
        self.__momentum = momentum
        
    def output(self, inputs):
        self._inputs = inputs
        
        E = inputs * self.__weights
        
        self.__out = numpy.sum(E) + self.__bias
        result = self.__function(self.__out)
        
        return result
        
    def update_delta(self):
        
        self.__Deltas = self.__Deltas +  self.__n * self.__delta * self._inputs
#         self.__Deltas = self.__Deltas + self.__momentum * self.__Deltas
        
        self.__Delta_bias = self.__Delta_bias + self.__n * self.__delta
        
    def update(self):
        self.__weights = self.__weights + self.__Deltas       
            
        self.__bias = self.__bias + self.__Delta_bias
        
        # limpa os Deltas
        
        self.__Deltas = numpy.zeros(len(self.__weights))
        self.__Delta_bias = 0
        
    def get_weight_n(self, n):
        return self.__weights[n]
    
    # encapsulamento
    
    def get_weights(self):
        return self.__weights
    
    def get_bias(self):
        return self.__bias
    
    def get_delta(self):
        return self.__delta
    
    def set_delta(self, delta):
        self.__delta = delta
        
    def get_out(self):
        return self.__out
    