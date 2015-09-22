'''
Created on 15/09/2015

@author: renan
'''

from ga.Genetic_Algorithm import Genetic_Algorithm
from model.MLP import MLP
from examples.xor import sig, dsig
from model.File import File

import numpy

def get_data():
    
    def f(x):
        result = numpy.zeros(3)
        
        if x == 'Iris-setosa':
            result[0] = 1
            
        elif x == 'Iris-versicolor':
            result[1] = 1
            
        elif x == 'Iris-virginica':
            result[2] = 1
        else:
            raise ValueError
        
        return result
    
    data = File(u'../../../data')
    data.extract()
    
    data.shuffle()
    data.convertClass(function=f, length=3)
    
    return data.allInputs, data.allCorrect

if __name__ == '__main__':
    
    train, target = get_data()
    
    mlp = MLP(function = sig,
              dfunction=dsig,
              features = 4,
              topology = [2,3])
    
    best_mlp = mlp
    
    def fenotype(individual):
        genotype = individual.get_genotype()
        
        pos = 0
        
        for i, layer in enumerate(mlp.get_layers()):
            for j, neuron in enumerate(layer.get_neurons()):
                
                end_neuron_position = pos + len(neuron.get_weights())
                
                try:
                    neuron.get_weights()[:] = genotype[pos: end_neuron_position]
                except ValueError:
                    print i, j
                    return None
                                                        
                neuron.set_bias(genotype[end_neuron_position])
                
                pos = end_neuron_position + 1
        
        len_train = len(train)
            
        accept = 0
                
        for inputs, correct in zip(train, target):
            output = mlp.output(inputs)
            
            if numpy.argmax(output) == numpy.argmax(correct):
                accept += 1.0/len_train
                
        return accept
    
    ga = Genetic_Algorithm(individuals = 100, 
                           mutation_rate = 0.1,
                           # (4 caracteristicas + 1 do bias) * 2 neuronios = 10
                           # (2 saidas de neuronio + 1 do bias) * 3 neuronios de saida = 9
                           # total de pesos 10+9 = 19
                           genotype = 19, 
                           fenotype = fenotype, 
                           genotype_type = Genetic_Algorithm.TYPE_FLOAT)
    
#     ga.print_individuals()
    ga.execute(generations = 100)
#     ga.print_individuals()
    print "Best Individual for all generations", ga.get_best_individual().get_genotype(), ga.get_best_individual_fenotype()
    
    genotype = ga.get_best_individual().get_genotype()
        
    pos = 0
    
    for i, layer in enumerate(mlp.get_layers()):
        for j, neuron in enumerate(layer.get_neurons()):
            
            end_neuron_position = pos + len(neuron.get_weights())
            
            neuron.get_weights()[:] = genotype[pos: end_neuron_position]
                                                    
            neuron.set_bias(genotype[end_neuron_position])
            
            pos = end_neuron_position + 1
            
    mlp.get_weights('pesos_ga_iris.mlp')
    