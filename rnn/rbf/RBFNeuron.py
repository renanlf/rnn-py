'''
Created on 22/09/2015

@author: renan
'''

class RBFNeuron(object):
    '''
    Esta classe representa um neurônio da camada RBF de uma rede neural
    '''


    def __init__(self, centroid, radius):
        '''
        Constructor
        '''
        self.__radius = radius
        self.__centroid = centroid
        
    def get_radius(self):
        return self.__radius
    
    def set_radius(self, radius):
        self.__radius = radius
        
    def get_centroid(self):
        return self.__centroid