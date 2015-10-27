# encoding: utf-8
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
        
        # carregando variaveis
        
        X = self.__data
        X_T = X.T
        
        #calculando a media
        mean_data = numpy.mean(X_T, axis=1)
        
        # subtraindo a media dos atributos para centralizar os dados
        for i, pattern in enumerate(X):
            X[i] = pattern - mean_data
        
        # como os dados estão centralizados obtenho a matriz de covariancia desta forma
        C = X_T.dot(X)
        
        #calcula os autovalores(values) e autovetores(vectors)
        values, vectors = numpy.linalg.eig(C)
        
        
        #######################################################################
        ############calculos utilizados para ordenar os autovetores############
        #######################################################################
        
        len_E = len(vectors)
        len_new_E = 0
        new_vectors = vectors
        
        while len_new_E < len_E:
            index = numpy.argmax(values)
            values[index] = -10000
            
            new_vectors[:, len_new_E] = vectors[:, index]
            len_new_E += 1
            
        #########################################################################
        
        #pega somente os dim primeiros autovetores
        
        new_vectors = new_vectors[:,0:dim]
        
        #calcula os novos valores        
        Z = X.dot(new_vectors)
        
        return Z