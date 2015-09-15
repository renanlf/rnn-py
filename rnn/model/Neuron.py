'''
Created on 18/08/2015

@author: renan
'''
import numpy
import warnings
from Model.NeuralNetworkError import NeuralNetworkError

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
            self.weights = numpy.float128(numpy.random.uniform(-0.5,0.5,weights))
            self.deltas = numpy.zeros(weights)
        else:
            self.weights = weights
            self.deltas = numpy.zeros(len(weights))
            
        self.function = function
        self.n = n
        self.bias = numpy.float128(numpy.random.uniform(-0.5,0.5,1))
        self.deltaBias = 0
        self.sigma = 0
        
        self.momentum = momentum
        
    def output(self, inputs):
        self.inputs = inputs
        try:
            E = inputs * self.weights
        except RuntimeWarning:
            print inputs, self.weights
        
        self.a = numpy.sum(E) + self.bias
        result = self.function(self.a)
        
        return result
        
    def updateDelta(self):
        
        self.deltas = self.deltas +  self.n * self.sigma * self.inputs
#         self.deltas = self.deltas + self.momentum * self.deltas
        
        self.deltaBias = self.deltaBias + self.n * self.sigma
        
    def update(self):
        self.weights = self.weights + self.deltas       
            
        self.bias = self.bias + self.deltaBias
        
        self.deltas = numpy.zeros(len(self.weights))
        self.deltaBias = 0
        
    def getWeight(self, n):
        return self.weights[n]
    