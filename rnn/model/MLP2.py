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
            layer.add_neuron(Neuron(weights=weights, function=function, n=n, momentum=momentum))
            i += 1
            
        self.__layers.append(layer)
        
    def output(self, inputs):
        inp = inputs
        
        for layer in self.__layers:
            out = layer.output(inp)
            inp = out
            
        return out
    
    def __train(self, inputs, correct, log=False):
            
        index_last_layer = len(self.__layers) - 1
        last_layer = self.__layers[index_last_layer]
        
        out = self.output(inputs)
        if log:
            print out, correct
            
        # calcula erros da camada de saida
        for i, neuron in enumerate(last_layer.get_neurons()):
            output_neuron = out[i]
            correctNeuron = correct[i]
            
            neuron.set_delta(self.__dfunction(neuron.get_out()) * (correctNeuron - output_neuron))
            neuron.update_delta()
            
        # cria uma lista com os indices das camadas ocultas            
        index_layers = range(0, index_last_layer)
        index_layers.reverse()
            
        # para cada camada...
        for l in index_layers:
            layer = self.__layers[l]
            next_layer = self.__layers[l + 1]
                
            # para cada neuronio...
            for i, neuron in enumerate(layer.get_neurons()):
                sum_next_layer_deltas = 0
                
                # para cada neuronio da camada __out frente...
                for next_layer_neuron in next_layer.get_neurons():
                    
                    # calcule os seus sigmas com os seus pesos de entrada
                    # utilizando o indice i estou pegando exatamente
                    # o peso que sai do neuron e entra no next_layer_neuron
                    sum_next_layer_deltas += next_layer_neuron.get_delta() * next_layer_neuron.get_weights()[i]
                        
                output_neuron = neuron.output(neuron._inputs)
                neuron.set_delta(self.__dfunction(neuron.get_out()) * sum_next_layer_deltas)
                neuron.update_delta()
                
        return out, correct
        
    def __update(self):
        for layer in self.__layers:
            for neuron in layer.get_neurons():
                neuron.update()
    
    def train_data(self, train, target, epochs = 1, log=False):
        result = numpy.zeros(epochs)
        len_all_inputs = len(train)
        
        for epoch in range(0, epochs):
            for i, inputs in enumerate(train):
                correct = target[i]
                
                out, correct = self.__train(inputs, correct, log)
                
                result[epoch] = result[epoch] + ((correct - out)**2)/len_all_inputs
                
            self.__update()
        
        return result                
        
    def get_weights(self, path=''):
        string = ''
        for i, layer in enumerate(self.__layers):
            string += 'Layer' + str(i + 1)
            
            for j, neuron in enumerate(layer.get_neurons()):
                string += '\n   Neuron'+ str(j+1) + ':'
                
                for k, weight in enumerate(neuron.get_weights()):
                    if k > 0:
                        string += ','
                    string += str(weight)
                    
                string += ','+str(neuron.get_bias())
                    
            string += '\n'

        
        if path != '':
            f = open(path, "w")
            f.write(string)
            
            f.close()
            
        return string
    
    def get_layers(self):
        return self.__layers
