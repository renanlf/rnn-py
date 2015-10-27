# encoding: utf-8
'''
Created on 22/09/2015

@author: renan
'''
import numpy

from rbf.RBFNeuron import RBFNeuron

class DDA(object):
    '''
    classdocs
    '''

    def train(data, target, rbf, theta_plus=.4, theta_minus=.1):
        i = 0
        for pattern, label in zip(data, target):
            print i
            i += 1
            label_rbfneurons = rbf.get_rbf_for_label(label)
            
            #somente o primeiro é infinito!!
            
            if len(label_rbfneurons) > 0 :
            
                #só os neuronios do label
                for rbf_neuron in label_rbfneurons:
                    if rbf_neuron.output(pattern) >= theta_plus:
                        
                        rbf_neuron.set_weight(rbf_neuron.get_weight() + 1.0)
                        
                    else:
                        rbf_neuron = RBFNeuron(centroid=pattern, label=label)
                        rbf_neuron.set_radius(numpy.inf)
                        
                        rbf_neuron.set_weight(1.0)
                        
                        max_radius = 0
                        
                        for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
                            centroid = other_label_rbf_neuron.get_centroid()
                            
                            out = rbf_neuron.output(t=centroid)
                            
                            distance = numpy.sum(rbf_neuron.get_centroid() - centroid)**2
                                
                            rd = (distance/(numpy.log(theta_minus))**2) + 1
                            
                            if rd > max_radius:
                                max_radius = rd
                                
                        rbf_neuron.set_radius(max_radius)
                                
                        rbf.get_rbf_neurons().append(rbf_neuron)
                                
            else:
                rbf_neuron = RBFNeuron(centroid=pattern, label=label)
                rbf_neuron.set_radius(numpy.inf)
                        
                rbf_neuron.set_weight(1.0)
                
                max_radius = 0
                        
                for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
                    centroid = other_label_rbf_neuron.get_centroid()
                            
                    distance = numpy.sum(rbf_neuron.get_centroid() - centroid)**2
                                
                    rd = (distance/(numpy.log(theta_minus))**2) + 1
                            
                    if rd > max_radius:
                        max_radius = rd
                                
                rbf_neuron.set_radius(max_radius)
                        
                rbf.get_rbf_neurons().append(rbf_neuron)
                
                rbf.add_neuron(label)
                print "append neuron for", label
            
            for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
                centroid = other_label_rbf_neuron.get_centroid()
                            
                distance = numpy.sum(centroid - pattern)**2
                                
                other_label_rbf_neuron.set_radius((distance/(-numpy.log(theta_minus))**0.5) + 1)
                
                        
                        
                    
    
    train = staticmethod(train)