'''
Created on 20/08/2015

@author: renan
'''
import numpy

class File(object):
    '''
    classdocs
    '''
    path = ''
    allInputs = []
    allClass = []
    allCorrect = []

    def __init__(self, path):
        '''
        Constructor
        '''
        self.path = path
    
    def extract(self, haveClass=True):
        f = open(self.path, "r")
        lines = f.readlines()
        firstLine = lines[0]
        firstLine = firstLine.replace("\n", "")
        columns = firstLine.split(",")
        
        if haveClass:
            clazz = columns[len(columns)-1]
            columns.remove(clazz)

        self.allInputs = numpy.ndarray([len(lines),len(columns)])
        self.allInputs[0,:] = columns
        
        self.allClass = []
        if haveClass:
            self.allClass.append(clazz)
        for i in range(1,len(lines)):
            line = lines[i]
            line = line.replace("\n","")
            columns = line.split(",")
            
            if haveClass:
                clazz = columns[len(columns)-1]
                columns.remove(clazz)

            self.allInputs[i,:] = columns
            if haveClass:
                self.allClass.append(clazz)
                
    def convertClass(self, function, length=1):
        self.allCorrect = numpy.ndarray([len(self.allClass),length])
        
        for i, clazz in enumerate(self.allClass):
            result = function(clazz)
            self.allCorrect[i,:] = result
            
    def shuffle(self):        
        indexes = numpy.arange(len(self.allInputs))
        numpy.random.shuffle(indexes)
        
        newAllInputs = numpy.ndarray([len(self.allInputs), len(self.allInputs[0])])
        newAllClass  = []
        
        for i, index in zip(range(0, len(self.allInputs)), indexes):
            
            newAllInputs[i,:] = self.allInputs[index,:]
            newAllClass.append(self.allClass[index])
                    
            
        self.allInputs = newAllInputs
        self.allClass  = newAllClass
        
    def split(self, rate=0.5):
        lenTrain = numpy.round(rate * len(self.allInputs))
        
        splittedInputs = numpy.split(self.allInputs, [lenTrain])
        splittedCorrect = numpy.split(self.allCorrect, [lenTrain])
        
        return splittedInputs[0], splittedCorrect[0], splittedInputs[1], splittedCorrect[1]
            
            
        