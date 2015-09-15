'''
Created on 18/08/2015

@author: renan
'''
import numpy
from model.Layer import Layer
from model.Neuron import Neuron

class MLP(object):
    '''
    classdocs
    '''
    __layers = []
    __dfunction = 0
    __lenOutput = 0
    def __init__(self, function, dfunction, n, features=1, topology=[], momentum = 0.1):
        '''
        Constructor
        '''
        self.__dfunction = dfunction
        self.__lenOutput = topology[-1]
        for i, neuronsLayer in enumerate(topology):
            if i == 0:
                self.__createLayer(neurons=neuronsLayer, weights=features, function=function, n=n, momentum=momentum)
            else:
                self.__createLayer(neurons=neuronsLayer, weights=topology[i-1], function=function, n=n, momentum=momentum)
    
    def addLayer(self, layer):
        self.__layers.append(layer)
        
    def __createLayer(self, neurons, weights, function, n, momentum=0.1):
        layer = Layer()
        
        i = 0
        while i < neurons:
            layer._addNeuron(Neuron(weights=weights, function=function, n=n, momentum=momentum))
            i += 1
            
        self.__layers.append(layer)
        
    def output(self, inputs):
        inp = inputs
        
        for layer in self.__layers:
            out = layer._output(inp)
            inp = out
            
        return out
    
    def __train(self, inputs, correct, log=False):
            
        indexLastLayer = len(self.__layers) - 1
        lastLayer = self.__layers[indexLastLayer]
        
        out = self.output(inputs)
        if log:
            print out, correct
            
        # calcula erros da camada de saida
        for i, neuron in enumerate(lastLayer._neurons):
            outNeuron = out[i]
            correctNeuron = correct[i]
            
            neuron._sigma =  self.__dfunction(neuron._a) * (correctNeuron - outNeuron)
            neuron._updateDelta()
            
        # cria uma lista com os indices das camadas ocultas            
        indexLayers = range(0, indexLastLayer)
        indexLayers.reverse()
            
        # para cada camada...
        for l in indexLayers:
            layer = self.__layers[l]
            nextLayer = self.__layers[l + 1]
                
            # para cada neuronio...
            for i, neuron in enumerate(layer._neurons):
                sumNextLayerSigmas = 0
                
                # para cada neuronio da camada _a frente...
                for nextLayerNeuron in nextLayer._neurons:
                    
                    # calcule os seus sigmas com os seus pesos de entrada
                    # utilizando o indice i estou pegando exatamente
                    # o peso que sai do neuron e entra no nextLayerNeuron
                    sumNextLayerSigmas += nextLayerNeuron._sigma * nextLayerNeuron._weights[i]
                        
                outNeuron = neuron._output(neuron._inputs)
                neuron._sigma = self.__dfunction(neuron._a) * sumNextLayerSigmas
                neuron._updateDelta()
                
        return out, correct
        
    def __updateAllNeurons(self):
        for layer in self.__layers:
            for neuron in layer._neurons:
                neuron._update()
    
    def trainData(self, allInputs, allCorrect, epochs = 1, log=False):
        result = numpy.zeros(epochs)
        lenAllInputs = len(allInputs)
        
        for epoch in range(0, epochs):
            for i, inputs in enumerate(allInputs):
                correct = allCorrect[i]
                
                out, correct = self.__train(inputs, correct, log)
                
                result[epoch] = result[epoch] + ((correct - out)**2)/lenAllInputs
                
            self.__updateAllNeurons()
        
        return result
    
    def updateN(self, n):
        for layer in self.__layers:
            for neuron in layer.neurons:
                neuron.n = n
                
        
    def getWeights(self, path=''):
        string = ''
        for i, layer in enumerate(self.__layers):
            string += 'Layer' + str(i + 1)
            
            for j, neuron in enumerate(layer._neurons):
                string += '\n   Neuron'+ str(j+1) + ':'
                
                for k, weight in enumerate(neuron._weights):
                    if k > 0:
                        string += ','
                    string += str(weight)
                    
                string += ','+str(neuron._bias)
                    
            string += '\n'

        
        if path != '':
            f = open(path, "w")
            f.write(string)
            
            f.close()
            
        return string
