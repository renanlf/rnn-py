'''
Created on 02/10/2015

@author: renan
'''
import numpy

class PCA(object):
    '''
    classdocs
    '''


    def __init__(self, data):
        '''
        Constructor
        '''
        self.__data = data
        
    def calculatePCA(self, dim=0):
    # verifica a quantidade de dimensoes a serem extraidas
    
        if dim == 0:
            dim = len(self.__data[0])
            
        #carrega a transposta dos dados de entrada...
        
        X = self.__data.T
        
        #matriz de covariancia
        C = numpy.cov(X)
        
        #calcula os autovalores(values) e autovetores(E)
        values, E = numpy.linalg.eig(C)
        
        
        #######################################################################
        ############calculos utilizados para ordenar os autovetores############
        #######################################################################
        
        len_E = len(E)
        len_new_E = 0
        new_E = E
        
        while len_new_E < len_E:
            index = numpy.argmax(values)
            values[index] = -10000
            
            new_E[:, len_new_E] = E[:, index]
            len_new_E += 1
            
        #########################################################################
        #pega somente os autovetores de maior dimensao
        
        new_E = new_E[0:dim,:]
        
        #calcula os novos valores
        
        Z = new_E.dot(X)
        
        return Z