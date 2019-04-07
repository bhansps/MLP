# coding=UTF-8
import math
import matplotlib.pyplot as plt
import copy

class MLP:
    def __init__(self, dataset, testSize):
        s = int(math.floor(len(dataset)*testSize))
        self.__dataTest = dataset[:s]
        self.__dataTraining = dataset[s:]
        self.__errorTest = []
        self.__errorTrainig = []
        self.__accuracyTest = []
        self.__accuracyTraining = []

    def setTheta(self, thetaInput, thetaHidden):
        self.__thetaI = copy.deepcopy(thetaInput) # [[thetaToHiddenLayer1], [thetaToHiddenLayer2], [thetaToHiddenLayer3], [thetaToHiddenLayer4]]
        self.__thetaH = copy.deepcopy(thetaHidden) # [[thetaToOutput1], [thetaToOutput2]]

    def setBias(self, biasInput, biasHidden):
        self.__biasI = copy.deepcopy(biasInput) # [biasHiddenLayer1, biasHiddenLayer2, biasHiddenLayer3, biasHiddenLayer4]
        self.__biasH = copy.deepcopy(biasHidden) # [biasOutput1, biasOutput2]

    def setAlpha(self, alpha):
        self.__alpha = alpha

    def __errorFunction(self, target, output):
        return ((target-output)**2)/2.0
    
    def __sigmoid(self, z):
        return 1.0/(1.0 + math.exp(-z))
    
    def __isPredictionTrue(self, target, output):
        for i in range(len(output)):
            if(output[i] > 0.5):
                o = 1
            else:
                o = 0
            if(o != target[i]):
                return False
        return True

    def train(self):
        error = 0.0
        accuracy = 0.0
        for data in self.__dataTraining:
            outH = [] # sigmoid net input
            for i in range(len(self.__thetaI)):
                outH.append(self.__sigmoid(sum(map(lambda x,y: x*y, data['input'], self.__thetaI[i])) + self.__biasI[i]))
            outO = [] # sigmoid net hidden layer
            for i in range(len(self.__thetaH)):
                outO.append(self.__sigmoid(sum(map(lambda x,y: x*y, outH, self.__thetaH[i])) + self.__biasH[i]))

             # update tetha & bias input
            for i in range(len(self.__thetaI)):
                deltaError = 0.0
                for k in range(len(outO)):
                    deltaError += (outO[k]-data['target'][k]) * outO[k]*(1-outO[k]) * self.__thetaH[k][i]
                for j in range(len(self.__thetaI[i])):
                    delta = deltaError * outH[i]*(1-outH[i]) * data['input'][j]
                    self.__thetaI[i][j] -= self.__alpha * float(delta)
                delta = deltaError * outH[i]*(1-outH[i])
                self.__biasI[i] -= self.__alpha * float(delta)

            # update tetha & bias hidden layer
            for i in range(len(self.__thetaH)):
                for j in range(len(self.__thetaH[i])):
                    delta = (outO[i]-data['target'][i]) * outO[i]*(1-outO[i]) * outH[j]
                    self.__thetaH[i][j] -= self.__alpha * float(delta)
                delta = (outO[i]-data['target'][i]) * outO[i]*(1-outO[i])
                self.__biasH[i] -= self.__alpha * float(delta)

            error += sum(map(lambda x,y: self.__errorFunction(x, y), data['target'], outO))
            if(self.__isPredictionTrue(data['target'], outO)):
                accuracy += 1

        self.__errorTrainig.append(error / len(self.__dataTraining))
        self.__accuracyTraining.append(accuracy / len(self.__dataTraining))

    def test(self):
        error = 0.0
        accuracy = 0.0
        for data in self.__dataTest:
            outH = [] # sigmoid net input
            for i in range(len(self.__thetaI)):
                outH.append(self.__sigmoid(sum(map(lambda x,y: x*y, data['input'], self.__thetaI[i])) + self.__biasI[i]))
            outO = [] # sigmoid net hidden layer
            for i in range(len(self.__thetaH)):
                outO.append(self.__sigmoid(sum(map(lambda x,y: x*y, outH, self.__thetaH[i])) + self.__biasH[i]))

            error += sum(map(lambda x,y: self.__errorFunction(x, y), data['target'], outO))
            if(self.__isPredictionTrue(data['target'], outO)):
                accuracy += 1

        self.__errorTest.append(error / len(self.__dataTest))
        self.__accuracyTest.append(accuracy / len(self.__dataTest))

    def showPlot(self):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.__errorTrainig, label='Training')
        plt.plot(self.__errorTest, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Total Error')
        plt.title(u'Error (α = {})'.format(self.__alpha), loc='left')
        plt.legend()
        plt.grid(True)

        plt.subplot(212)
        plt.plot(self.__accuracyTraining, label='Training')
        plt.plot(self.__accuracyTest, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Accuracy')
        plt.title(u'Accuracy (α = {})'.format(self.__alpha), loc='left')
        plt.legend()
        plt.grid(True)

        plt.show()
