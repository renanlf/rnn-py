'''
Created on 22/09/2015

@author: renan
'''
import numpy

alpha = numpy.float128(1.0)

def sig(x):
    return numpy.float128(1.0) / (numpy.float128(1.0) + numpy.exp(-x*alpha))
    
def dsig(x):
    return (numpy.exp(-x*alpha))/((numpy.float128(1.0) + numpy.exp(-x*alpha))**2)