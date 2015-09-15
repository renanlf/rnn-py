'''
Created on 18/08/2015

@author: renan
'''
import numpy
from Model.Layer import Layer
from Model.Neuron import Neuron
class MLP(object):
    '''
    classdocs
    '''
    layers = []
    error = 0
    dfunction = 0
    lenOutput = 0
    def __init__(self, function, dfunction, n, features=1, topology=[], momentum = 0.1):
        '''
        Constructor
        '''
        self.dfunction = dfunction
        self.lenOutput = topology[-1]
        for i, neuronsLayer in enumerate(topology):
            if i == 0:
                self.createLayer(neurons=neuronsLayer, weights=features, function=function, n=n, momentum=momentum)
            else:
                self.createLayer(neurons=neuronsLayer, weights=topology[i-1], function=function, n=n, momentum=momentum)
    
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def createLayer(self, neurons, weights, function, n, momentum=0.1):
        layer = Layer()
        
        i = 0
        while i < neurons:
            layer.addNeuron(Neuron(weights=weights, function=function, n=n, momentum=momentum))
            i += 1
            
        self.layers.append(layer)
        
    def output(self, inputs):
        inp = inputs
        
        for layer in self.layers:
            out = layer.output(inp)
            inp = out
            
        return out
    
    def train(self, inputs, correct, log=False):
            
        indexLastLayer = len(self.layers) - 1
        lastLayer = self.layers[indexLastLayer]
        
        out = self.output(inputs)
        if log:
            print out, correct
            
        # calcula erros da camada de saida
        for i, neuron in enumerate(lastLayer.neurons):
            outNeuron = out[i]
            correctNeuron = correct[i]
            
            neuron.sigma =  self.dfunction(neuron.a) * (correctNeuron - outNeuron)
            neuron.updateDelta()
            
        # cria uma lista com os indices das camadas ocultas            
        indexLayers = range(0, indexLastLayer)
        indexLayers.reverse()
            
        # para cada camada...
        for l in indexLayers:
            layer = self.layers[l]
            nextLayer = self.layers[l + 1]
                
            # para cada neuronio...
            for i, neuron in enumerate(layer.neurons):
                sumNextLayerSigmas = 0
                
                # para cada neuronio da camada a frente...
                for nextLayerNeuron in nextLayer.neurons:
                    
                    # calcule os seus sigmas com os seus pesos de entrada
                    # utilizando o indice i estou pegando exatamente
                    # o peso que sai do neuron e entra no nextLayerNeuron
                    sumNextLayerSigmas += nextLayerNeuron.sigma * nextLayerNeuron.weights[i]
                        
                outNeuron = neuron.output(neuron.inputs)
                neuron.sigma = self.dfunction(neuron.a) * sumNextLayerSigmas
                neuron.updateDelta()
                
        return out, correct
        
    def updateAllNeurons(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update()
                
    def getError(self, allInputs, allCorrect):
        error = 0
        
        for inputs, correct in zip(allInputs, allCorrect):
            out = self.output(inputs)
            error += (correct - out)**2
        
        return error/len(allInputs)
    
    def trainAllData(self, allInputs, allCorrect, log=False):
        error = 0
        for i, inputs in enumerate(allInputs):
            correct = allCorrect[i]
            
            out, correct = self.train(inputs, correct, log)
            
            error += (correct - out)**2
            
        self.updateAllNeurons()
        
        return error/len(allInputs)
    
    def updateN(self, n):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.n = n
                
        
    def getWeights(self, path=''):
        string = ''
        for i, layer in enumerate(self.layers):
            string += 'Layer' + str(i + 1)
            
            for j, neuron in enumerate(layer.neurons):
                string += '\n   Neuron'+ str(j+1) + ':'
                
                for k, weight in enumerate(neuron.weights):
                    if k > 0:
                        string += ','
                    string += str(weight)
                    
                string += ','+str(neuron.bias)
                    
            string += '\n'

        
        if path != '':
            f = open(path, "w")
            f.write(string)
            
            f.close()
            
        return string
