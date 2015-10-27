# encoding: utf-8
'''
Created on 27/10/2015

@author: renan
'''
from rbf.DDA import DDA
from rbf.RBF import RBF
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
    
    return data.split(rate=0.7)

if __name__ == '__main__':
    train, target, test, target_test = get_data()
    
    rbf = RBF(len_output_layer=3)
    
    DDA.train(data = train, target = target, rbf = rbf)
    
    neurons = rbf.get_rbf_neurons()
    
    for rbf_neuron in neurons:
        print rbf_neuron.get_centroid(), rbf_neuron.get_radius(), rbf_neuron.get_label()
    
        
    correct = 0.0
    for pattern, label in zip(test, target_test):
        out = rbf.output(pattern=pattern)
        
        if numpy.argmax(out) == numpy.argmax(label):
            correct += 1.0
        
    print correct/len(target_test)
        