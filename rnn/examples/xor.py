'''
Created on 15/09/2015

@author: renan
'''
import numpy

from model.MLP2 import MLP

def sig(x):
    return 1 / (1 + numpy.e**(-x))
    
def dsig(x):
    return (numpy.e**(-x))/((1 + numpy.e**(-x))**2)

def createData():
    inputs = numpy.ndarray((10,2))
    inputs[:,0] = numpy.random.randint(low=0, high=2, size=10)
    inputs[:,1] = numpy.random.randint(low=0, high=2, size=10)
    xor = numpy.ndarray((10,1))
    xor[:,0] = numpy.logical_xor(inputs[:,0], inputs[:,1])
    return inputs, xor

if __name__ == '__main__':
    inputs, xor = createData()
    
    mlp = MLP(function = sig, 
              dfunction=dsig,
              n = 0.1, 
              features = 2, 
              topology = [2,1], 
              momentum = 0.1)
    
    errors = mlp.trainData(allInputs = inputs, allCorrect = xor, epochs = 20)
    print errors
    print mlp.getWeights()