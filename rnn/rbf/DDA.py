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

    def train(self, data, target, rbf, theta_plus=0.4, theta_minus=0.1):
        for pattern, label in zip(data, target):
            #só os neuronios do label
            for rbf_neuron in rbf.get_rbf_for_label(label):
                if rbf_neuron.output(pattern) >= theta_plus:
                    
                    rbf_neuron.set_weight(rbf_neuron.get_weight())
                    
                else:
                    rbf_neuron = RBFNeuron(centroid=pattern, label=label)
                    radius = 0
                    
                    for other_label_rbf_neuron in rbf.get_rbf_for_not_label(label):
                        centroid = other_label_rbf_neuron.get_centroid()
                        
                        distance = numpy.sum(rbf_neuron.get_centroid() - centroid)
                        
                        
                    
    
    train = staticmethod(train)