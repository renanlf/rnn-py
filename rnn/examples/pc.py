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
    
#     data.shuffle()
    data.convertClass(function=f, length=3)
    
    return data.allInputs, data.allClass

if __name__ == '__main__':
    X, Y = get_data()
    
    pca = PCAr(data = X)
    
    Z = pca.calculatePCA(dim=2)
    
    sk_pca = PCA(n_components=2)
    
    Z2 = sk_pca.fit_transform(X)
    
    plots = []
    
    plots.append(pyplot.scatter(Z[0:50,0], Z[0:50,1], color='red'))
    plots.append(pyplot.scatter(Z[50:100,0], Z[50:100,1], color='blue'))
    plots.append(pyplot.scatter(Z[100:150,0], Z[100:150,1], color='green'))
    pyplot.legend(plots,['Iris-setosa','Iris-versicolor','Iris-virginica'])
    
#     pyplot.subplot(132)
#     pyplot.scatter(Z2[0:50,0], Z2[0:50,1], color='red')
#     pyplot.scatter(Z2[50:100,0], Z2[50:100,1], color='blue')
#     pyplot.scatter(Z2[100:150,0], Z2[100:150,1], color='green')
#     
#     pyplot.subplot(133)
#     pyplot.scatter(X[:,0], X[:,1])
    
    pyplot.show()