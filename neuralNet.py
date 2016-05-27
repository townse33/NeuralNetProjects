import numpy as np

def sigmoid(x): #sigmoid activation function
    return 1/(1+np.exp(-x))
def sigmoidPrime(x): #sigmoid derivative
    return x*(1-x)

class NeuralNet: #Generic class for defining neural networks

    def __init__(self,inputList,targetList,hiddenLayers=1,hiddenNeurons=False):

        #Hyperparameters
        self.inputNo = len(inputList) #Number of input nodes
        self.outputNo = len(targetList) #Number of output nodes
        
        self.bias = 1 #Neuron bias
        self.learnRate = 1 #Rate of learning for training

        self.inputL = inputList 
        self.targetL = targetList

        rangeList = inputList.copy()
        rangeList.extend(targetList) #Compare all values, both training and input

        self.minNum = min(rangeList) #Find minimum value
        self.maxNum = max(rangeList) #Find maximum value

        self.finalIn = list() 

        for inputNum in inputList:
            if self.maxNum - self.minNum != 0:
                self.finalIn.append((inputNum-self.minNum)/(self.maxNum-self.minNum)) #Normalise data
            else:
                self.finalIn.append(inputNum)
        
        self.hLayers = hiddenLayers #Number of hidden layers -> hLayers
        
        #If no. of neurons per hidden layer not given, we assume it's the no. of inputs
        if hiddenNeurons == False:
            self.hNeurons = self.inputNo
        else:
            self.hNeurons = hiddenNeurons
            
        self.startWeights = np.random.rand(self.inputNo,self.hNeurons) #Mersenne Twister PRNG for a matrix of dimensions: input size x hidden layer size
        self.endWeights = np.random.rand(self.hNeurons,self.outputNo) #PRNG matrix of dimensions: hidden layer size x output size
        
        self.weightList = list() #Used to store weight matrices between hidden layers
        self.aList = list() #Used to store activation matrices
        self.inList = list() #Used to store input matrices

        if self.hLayers > 1:
            for k in range(self.hLayers-1):
                self.weightList.append(np.random.rand(self.hNeurons,self.hNeurons)) #Generate hidden layer weight matrices

    def fowardProp(self): #Feed forward inputs through the network
        self.firstLayer = np.dot(self.finalIn,self.startWeights)+self.bias  #Matrix multiply input matrix by first weight matrix, add bias
        self.currentMatrix = sigmoid(self.firstLayer)#Apply sigmoid activation function to previous matrix
        for weightMatrix in self.weightList:
            self.currentMatrix = np.dot(self.currentMatrix,weightMatrix)+self.bias #Continue matrix multiplying through hidden layers
            self.inList.append(self.currentMatrix) #Add neuron inputs to the input list
            self.currentMatrix = sigmoid(self.currentMatrix)
            self.aList.append(self.currentMatrix) #Add activation values to activation list
        outList = np.dot(self.currentMatrix,self.endWeights)+self.bias #Final output values found using sigmoid
        self.inList.append(outList) #Output values before activation are treated as inputs
        outList = sigmoid(outList)
        #print(outList) #Print output, return output
        return outList

    def backProp(self): #Backprop algorithm using Stochastic Gradient Descent

        self.out = np.asarray(self.fowardProp()) #Convert lists to arrays, create list of all weight matrices
        self.inputA = np.asarray(self.finalIn)
        self.targetA = np.asarray(self.targetL)
        self.aMatrix = np.asarray(self.aList)
        self.wMatrix = self.weightList.copy()
        self.wMatrix.append(self.endWeights)
        self.wMatrix.insert(0,self.startWeights)

        deltaO = np.multiply(sigmoidPrime(self.out),(self.targetA-self.out)) #Hadamard product of output function deriv and error
        self.endWeights += self.learnRate * np.outer(self.aMatrix[-1].T,deltaO) #Adjust weights by learning rate, delta and activation values,
                                                                                #outer product is used as matrices should be in vector form here
        deltaN = deltaO #Variable used to hold previous delta calculation

        for k in range(self.hLayers-1): #Iterates between hidden layers
            deltaN = deltaN.dot(self.wMatrix[-k-1].T)*sigmoidPrime(self.aMatrix[-k-1]) #Apply chain rule to calculate error
            self.weightList[-k-1] += self.learnRate * np.dot(self.aMatrix[-k-1].T,deltaN) #Update weights by the found delta and synapses

        deltaI = deltaN.dot(self.wMatrix[1].T)*sigmoidPrime(self.aMatrix[0]) #Final delta
        self.startWeights += self.learnRate * np.dot(self.inputA.T,deltaI) #Updates starting weights


    def train(self,iterations=1): #Iterates backprop for given number of epochs

        for i in range(iterations):
            self.backProp()

    def changeValues(self,inputs,targets): #Used to change values for inputs and their target outputs (MUST be lists)
        self.inputL = inputs
        self.targetL = targets

        self.inputNo = len(self.inputL)
        self.outputNo = len(self.targetL)

        self.finalIn = self.inputL.copy()
            
test = NeuralNet([0.1,0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5,0.6],5)

test.train(1000)

print(test.fowardProp())
