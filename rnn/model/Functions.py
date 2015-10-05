'''
Created on 22/09/2015

@author: renan
'''
import numpy

alpha = 0.5

def sig(x):
    return 1 / (1 + numpy.e**(-x*alpha))
    
def dsig(x):
    return (numpy.e**(-x*alpha))/((1 + numpy.e**(-x*alpha))**2)