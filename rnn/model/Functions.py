'''
Created on 22/09/2015

@author: renan
'''
import numpy

def sig(x):
    return 1 / (1 + numpy.e**(-x))
    
def dsig(x):
    return (numpy.e**(-x))/((1 + numpy.e**(-x))**2)