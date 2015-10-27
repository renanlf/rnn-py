# encoding: utf-8
'''
Created on 22/09/2015

@author: renan
'''
import numpy

from rbf.RBFNeuron import RBFNeuron
from rbf.OutputNeuron import OutputNeuron

class DDA(object):
    '''
    classdocs
    '''

    def train(data, target, rbf, theta_plus=.4, theta_minus=.1, epochs=1):
        
        #criando primeiro neuronio rbf e o adicionando na rede
        first_neuron = RBFNeuron(centroid=data[0], label=target[0], radius=numpy.inf)
        first_neuron.set_weight(1.0)
        
        rbf.get_rbf_neurons().append(first_neuron)
        
        itered = 0
        
        while itered < epochs:
            itered += 1
            print itered            
        
            for pattern, label in zip(data[1:len(data)], target[1:len(target)]):
            
                label_rbfneurons = rbf.get_rbf_for_label(label)
                
                #somente o primeiro é infinito!!
                
                if len(label_rbfneurons) > 0 :
                
                    #só os neuronios do label
                    for rbf_neuron in label_rbfneurons:
                        if rbf_neuron.output(pattern) >= theta_plus:
                            
                            rbf_neuron.set_weight(rbf_neuron.get_weight() + 1.0)
                            
                        else:
                            DDA.__add_rbf_neuron(rbf, pattern, label, theta_minus)
                                    
                else:
                    DDA.__add_rbf_neuron(rbf, pattern, label, theta_minus)
                
                max_radius = DDA.__max_radius(rbf=rbf, rbf_centroid=pattern, label=label, theta_minus=theta_minus)
                        
                for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
                    other_label_rbf_neuron.set_radius(max_radius)
                    
    
    train = staticmethod(train)
    
    def __add_rbf_neuron(rbf, pattern, label, theta_minus):
        rbf_neuron = RBFNeuron(centroid=pattern, label=label)
                        
        rbf_neuron.set_weight(1.0)
                        
        max_radius = DDA.__max_radius(rbf, rbf_neuron.get_centroid(), label, theta_minus)
                                
        rbf_neuron.set_radius(max_radius)
                                
        rbf.get_rbf_neurons().append(rbf_neuron)
        
    __add_rbf_neuron = staticmethod(__add_rbf_neuron)
    
    def __max_radius(rbf, rbf_centroid, label, theta_minus):
        max_radius = 0
                        
        for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
            centroid = other_label_rbf_neuron.get_centroid()
                            
            distance = numpy.sum(rbf_centroid - centroid)**2
                                
            rd = (distance/(numpy.log(theta_minus))**2) + 0.01
                            
            if rd > max_radius:
                max_radius = rd
                
        return max_radius
    
    __max_radius = staticmethod(__max_radius)