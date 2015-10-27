'''
Created on 27/10/2015

@author: renan
'''

class OutputNeuron(object):
    '''
    classdocs
    '''


    def __init__(self, label):
        '''
        Constructor
        '''
        self.__label = label
        
    def output(self, rbf, pattern):
        up = 0
        down = 0
        
        for rbf_neuron in rbf.get_rbf_for_label(self.__label):
            up += rbf_neuron.get_weight() * rbf_neuron.output(pattern)
            down += rbf_neuron.get_weight()
            
        return up/down
            
    