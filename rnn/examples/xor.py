'''
Created on 15/09/2015

@author: renan
'''
import numpy

from model.MLP import MLP

def createData():
    inputs = numpy.ndarray((10,2))
    inputs[:,0] = numpy.random.randint(low=0, high=2, size=10)
    inputs[:,1] = numpy.random.randint(low=0, high=2, size=10)
    xor = numpy.ndarray((10,1))
    xor[:,0] = numpy.logical_xor(inputs[:,0], inputs[:,1])
    return inputs, xor

if __name__ == '__main__':
    inputs, xor = createData()
    
    mlp = MLP(n = 0.1,
              features = 2, 
              topology = [2,1])
    
    errors = mlp.train_data(train = inputs, target = xor, epochs = 20)
    print errors
#     print mlp.get_weights()