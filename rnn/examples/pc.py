'''
Created on 02/10/2015

@author: renan
'''

from pca.PCA import PCA as PCAr
from model.File import File
from matplotlib import pyplot
from sklearn.decomposition import PCA

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
    X, Y = get_data()
    
    pca = PCAr(data = X)
    
    Z = pca.calculatePCA(dim=2)
    
    sk_pca = PCA(n_components=2)
    
    Z2 = sk_pca.fit_transform(X)
    
    print len(Z)
    
    pyplot.subplot(131)
    pyplot.scatter(Z[0,:], Z[1,:])
    
    pyplot.subplot(132)
    pyplot.scatter(Z2[:,0], Z2[:,1])
    
    pyplot.subplot(133)
    pyplot.scatter(X[:,0], X[:,1])
    
    pyplot.show()