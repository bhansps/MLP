import MLP
import random

if __name__=='__main__':
    f = open('data.csv', 'r')
    dataRaw = f.read()
    f.close()

    dataRaw = [[x for x in y.split(',')] for y in dataRaw.split('\n')]
    dataset = []
    for i in range(50):
        for j in range(3):
            inx = (50*j+i)%150
            temp = {}
            temp['input'] = [dataRaw[inx][0], dataRaw[inx][1], dataRaw[inx][2], dataRaw[inx][3]]
            temp['input'] = map(lambda x: float(x), temp['input'])
            temp['name'] = dataRaw[inx][4]
            if(temp['name'] == 'Iris-setosa'):
                temp['target'] = [0, 0]
            elif(temp['name'] == 'Iris-versicolor'):
                temp['target'] = [1, 0]
            elif(temp['name'] == 'Iris-virginica'):
                temp['target'] = [0, 1]
            dataset.append(temp)
    random.seed(5)
    random.shuffle(dataset)

    ML_1 = MLP.MLP(dataset, 0.3)
    ML_2 = MLP.MLP(dataset, 0.3)
    thetaInput = [[0.3, 0.7, 0.1, 0.85], [0.6, 0.3, 0.75, 0.5], [0.2, 0.85, 0.3, 0.7], [0.55, 0.4, 0.65, 0.2]]
    thetaHidden = [[0.4, 0.7, 0.2, 0.65], [0.75, 0.35, 0.6, 0.4]]
    # thetaInput = [[0.5]*4]*4
    # thetaHidden = [[0.5]*4]*2
    ML_1.setTheta(thetaInput, thetaHidden)


    biasInput = [0.6, 0.4, 0.7, 0.35]
    biasHidden = [0.5, 0.5]
    # biasInput = [0.5]*4
    # biasHidden = [0.5]*2
    ML_1.setBias(biasInput, biasHidden)

    epoch = 500
    ML_1.setAlpha(0.1)

    for _ in range(epoch):
        ML_1.train()
        ML_1.test()
    ML_1.showPlot()

    thetaInput = [[0.3, 0.7, 0.1, 0.85], [0.6, 0.3, 0.75, 0.5], [0.2, 0.85, 0.3, 0.7], [0.55, 0.4, 0.65, 0.2]]
    thetaHidden = [[0.4, 0.7, 0.2, 0.65], [0.75, 0.35, 0.6, 0.4]]
    ML_2.setTheta(thetaInput, thetaHidden)

    biasInput = [0.6, 0.4, 0.7, 0.35]
    biasHidden = [0.5, 0.5]
    ML_2.setBias(biasInput, biasHidden)

    ML_2.setAlpha(0.8)
    
    epoch = 500
    for _ in range(epoch):
        ML_2.train()
        ML_2.test()
    ML_2.showPlot()
