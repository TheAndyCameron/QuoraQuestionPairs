import math as m
import numpy as np
import random as rand

class mlpnode:
    eta = 0.2

    def __init__(self):
        self.inputnodes = []
        self.inputweights = []
    
        self.outputnodes = []
    
        self.forwardValue = 0
        self.backwardValue = 0 #Sensitivity to be backprop-ed

    
    def sigmoid(self, v):
        #return np.tanh(v)
        return 1/(1+m.exp(-v))
        
    def sigmoidDiff(self, v):
        s = self.sigmoid(v)
        #return 1-m.pow(s,2)
        return s*(1-s)
        
    def feedforward(self):
        self.forwardValue = self.sigmoid(self.sumWeightedInputs())
    
    def sumWeightedInputs(self):
        newfwvalue = 0
        for i in range(len(self.inputweights)):
            newfwvalue += self.inputnodes[i].forwardValue*self.inputweights[i]
        return newfwvalue
    
    
    
    def initialBackprop(self, output, target):
        #Calculate sensitivity
        #Look up "Pattern Classification - Duda et.al." for details
        netk = self.sumWeightedInputs()
        sensitivity = (target - output)*self.sigmoidDiff(netk)
        self.backwardValue = sensitivity #Allow previous layers access
        
        #now changes on weight
        for i in range(len(self.inputnodes)):
            xi = self.inputnodes[i].forwardValue
            weightChange = self.eta*xi*sensitivity
            self.inputweights[i] += weightChange
    
    def backprop(self):
        #First we calculate the sensitivity of this node
        #Look up "Pattern Classification - Duda et.al." for details
        netj = self.sumWeightedInputs()
        
        sumWeightedSensitivities = 0
        for n in self.outputnodes:
            val = n.getWeightForNodeWeight(self)*n.backwardValue
            sumWeightedSensitivities += val
    
        sensitivity = self.sigmoidDiff(netj)*sumWeightedSensitivities
        self.backwardValue = sensitivity #Allow previous layers access
        
        #now changes on weight
        for i in range(len(self.inputnodes)):
            xi = self.inputnodes[i].forwardValue
            weightChange = self.eta*xi*sensitivity
            self.inputweights[i] += weightChange
        
        #done!
    
    
    def addInNode(self, node):
        self.inputnodes.append(node)
        self.inputweights.append(rand.uniform(-0.9,0.9))
        #print("node added")
        
    def addOutNode(self, node):
        self.outputnodes.append(node)
    
    def getWeightForNodeWeight(self, node):
        index = self.inputnodes.index(node)
        if index > -1:
            return self.inputweights[index]
        else:
            return 0
    

class inputNode:
    def __init__(self, fv=0):
        self.forwardValue = fv
    
    def setValue(self, value):
        self.forwardValue = value
		
    def backprop(self):
        #do nothing
        _ = "_"


class MLP:
    
    layers = []
    
    def __init__(self, netStructure, eta=0.01):
        #set learning rate
        mlpnode.eta = eta

        #netStructure is an array of ints > 0
        
        for i in range(len(netStructure)):
            self.layers.append([])
            if i==0:
                for n in range(netStructure[i]):
                    self.layers[i].append(inputNode())
                    
            else:
                for n in range(netStructure[i]): 
                    self.layers[i].append(mlpnode())

        #make Connections
        #inputs
        for i in range(1, len(self.layers)):
            for node in self.layers[i]:
                for inNode in self.layers[i-1]:
                    node.addInNode(inNode)
                node.addInNode(inputNode(fv=1))
					
        #outputs
        for i in range(1, len(self.layers)-1):
            for node in self.layers[i]:
                for outNode in self.layers[i+1]:
                    node.addOutNode(outNode)

    def feedForward(self, inputArr):
        #set inputs
        for i in range(len(self.layers[0])):
            self.layers[0][i].forwardValue = inputArr[i]

        #iterate through layers
        for i in range(1, len(self.layers)):
            for node in self.layers[i]:
                node.feedforward()
                
        #get outputs
        outs = []
        for node in self.layers[-1]:
            outs.append(node.forwardValue)
        return outs

    def backProp(self, outputArr, targetArr):
        #initial bp on final layer
        for i in range(len(self.layers[-1])):
            self.layers[-1][i].initialBackprop(outputArr[i], targetArr[i])
        
        #iterate bp though previous layers
        for i in reversed(range(1,len(self.layers)-1)):
            for node in self.layers[i]:
                node.backprop()


    def dumpweights(self):
        for i in range(1,len(self.layers)):
            print("layer " + str(i))
            for n in self.layers[i]:
                print(n.inputweights)




#TESTING



