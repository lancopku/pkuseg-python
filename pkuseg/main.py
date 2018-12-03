from .feature import *
from .dataformat import *
from .toolbox import *
from .resSummarize import write as reswrite
from .resSummarize import summarize
import sys
import os
from .inference import *
from .config import Config
import time
from .ProcessData import tocrfoutput

def run(config=None):
    if config is None:
        config=Config()
    if config.runMode.find('train')>=0:
        trainFeature = Feature(config, config.trainFile, 'train')
        testFeature = Feature(config, config.testFile, 'test')
    else:
        testFeature = Feature(config, config.readFile, 'test')

    if config.formatConvert:
        df = dataFormat(config)
        df.convert()

    config.globalCheck()
    
    # TODO
    # directoryCheck()

    config.swLog = open(config.outDir + config.fLog, 'w')
    config.swResRaw = open(config.outDir + config.fResRaw, 'w')
    config.swTune = open(config.outDir + config.fTune, 'w')

    if config.runMode.find('tune')>=0:
        print('\nstart tune...')
        config.swLog.write('\nstart tune...\n')
        tuneStochasticOptimizer(config)
    elif config.runMode.find('train')>=0:
        print('\nstart training...')
        config.swLog.write('\nstart training...\n')
        if config.runMode.find('rich')>=0:
            richEdge.train()
        else:
            train(config) 
    elif config.runMode.find('test')>=0:
        config.swLog.write('\nstart testing...\n')
        if config.runMode.find('rich')>=0:
            richEdge.test()
        else:
            test(config)
        tocrfoutput(config, config.outFolder+'outputTag.txt', config.outputFile, config.tempFile+'/test.raw.txt')
    elif config.rumMode.find('cv')>=0:
        print('\nstart cross validation')
        config.swLog.write('\nstart cross validation\n')
        crossValidation(config)
    else:
        raise Exception('error')

    config.swLog.close()
    config.swResRaw.close()
    config.swTune.close()
    
    if config.runMode.find('train')>=0:
        summarize(config)

    #ClearDirectory(config.tempFile)

    print('finished.')

def train(config):
    print('\nreading training & test data...')
    config.swLog.write('\nreading training & test data...\n')
    if config.runMode.find('tune')>=0:
        origX = dataSet(config.fFeatureTrain, config.fGoldTrain)
        X = dataSet()
        XX = dataSet()
        dataSplit(origX, config.tuneSplit, X, XX)
    else:
        X = dataSet(config.fFeatureTrain, config.fGoldTrain)
        XX = dataSet(config.fFeatureTest, config.fGoldTest)
        dataSizeScale(config, X)
    print('done! train/test data sizes: {}/{}'.format(len(X), len(XX)))
    config.swLog.write('done! train/test data sizes: {}/{}\n'.format(len(X), len(XX)))
    for r in config.regList:
        config.reg = r
        config.swLog.write('\nr: '+str(r)+'\n')
        print('\nr: '+str(r))
        if config.rawResWrite:
            config.swResRaw.write('\n%r: '+str(r)+'\n')
        tb = toolbox(config, X, True)
        score = basicTrain(config, XX, tb)
        reswrite(config)
        if config.save == 1:
            tb.Model.save(config.fModel)
        return score

def test(config):
    config.swLog.write('reading test data...\n')
    XX = dataSet(config.fFeatureTest, config.fGoldTest)
    print('test data size: {}'.format(len(XX)))
    config.swLog.write('Done! test data size: {}\n'.format(len(XX)))
    tb = toolbox(config, XX, False)
    scorelist = tb.test(XX, 0)
        
def crossValidation(config):
    print('reading cross validation data...')
    config.swLog.write('reading cross validation data...\n')
    XList = []
    XXList = []
    loadDataForCV(config, XList, XXList)
    for r in config.regList:
        config.swLog.write('\ncross validation. r={}\n'.format(r))
        print('\ncross validation. r={}'.format(r))
        if config.rawResWrite:
            config.swResRaw.write('% cross validation. r={}'.format(r))
        for i in range(config.nCV):
            config.swLog.write('\n#validation={}\n'.format(i+1))
            print('\n#validation={}'.format(i+1))
            if config.rawResWrite:
                config.swResRaw.write('\n#validation={}\n'.format(i+1))
            config.reg = r
            Xi = XList[i]
            if config.runMode.find('rich')>=0:
                tb = toolboxRich(Xi)
                basicTrain(config, XXList[i], tb)
            else:
                tb = toolbox(config, Xi)
                basicTrain(config, XXList[i], tb)
            reswrite(config)
            if config.rawResWrite:
                config.swResRaw.write('\n')
        if config.rawResWrite:
            config.swResRaw.write('\n')

def tuneStochasticOptimizer(config):
    if config.modelOptimizer.endswith('sgd') or config.modelOptimizer.endswith('sgder') or config.modelOptimizer.endswith('adf'):
        origTtlIter = config.ttlIter
        origRegList = config.regList
        config.ttlIter = config.iterTuneStoch
        config.regList = []
        config.regList.append(1)
        config.rawResWrite = False
        rates = [0.1, 0.05, 0.01, 0.005]
        bestRate = -1
        bestScore = 0.
        for im in rates:
            config.rate0 = im
            score = reinitTrain(config)
            strlog = 'reg={}  rate0={} --> {}={}%'.format(config.regList[0], im, conifig.metric, '%.2f'%score)
            config.swTune.write(strlog+'\n')
            config.swLog.write(strlog+'\n')
            print(strlog)
            if score > bestScore:
                bestScore = score
                bestRate = im
        config.rate0 = bestRate
        bestScore = 0
        bestReg = -1
        regs = [0, 1, 2, 5, 10]
        config.swTune.write('\n')
        for im in regs:
            config.regList.clear()
            config.regList.append(im)
            score = reinitTrain(config)
            strlog = "reg={}  rate0={} --> {}={}%".format(config.regList[0], config.rate0, config.metric, '%.2f'%score)
            config.swTune.write(strlog+'\n')
            config.swLog.write(strlog+'\n')
            print(strlog)
            if score > bestScore:
                bestScore = score
                bestReg = im
        config.reg = bestReg
        config.swTune.write('\nconclusion: best-rate0={}, best-reg={}'.format(config.rate0, config.reg))
        config.ttlIter = origTtlIter
        config.regList = oriRegList
        config.rawResWrite = True
        print('done')
    else:
        print('no need tuning for non-stochastic optimizer! done.')
        config.swLog.write('no need tuning for non-stochastic optimizer! done.\n')

def reinitTrain(config):
    config.reinitGlobal()
    score = 0
    if config.runMode.find('rich')>=0:
        score = richEdge.train()
    else:
        score = train()
    return score

def basicTrain(config, XTest, tb):
    config.reinitGlobal()
    score = 0
    if config.modelOptimizer.endswith('bfgs'):
        config.tb = tb
        config.XX = XTest
        tb.train()
        score = config.scoreListList[-1][0]
    else:
        for i in range(config.ttlIter):
            config.glbIter += 1
            time_s = time.time()
            err = tb.train()
            time_t = time.time() - time_s
            config.timeList.append(time_t)
            config.errList.append(err)
            config.diffList.append(config.diff)
            scoreList = tb.test(XTest, i)
            config.scoreListList.append(scoreList)

            logstr = 'iter{}  diff={}  train-time(sec)={}  {}={}%'.format(config.glbIter, '%.2e'%config.diff, '%.2f'%time_t, config.metric, '%.2f'%score)
            config.swLog.write(logstr+'\n')
            config.swLog.write('------------------------------------------------\n')
            config.swLog.flush()
            print(logstr)
    return score

def dataSizeScale(config, X):
    XX = dataSet()
    XX.setDataInfo(X)
    for im in X:
        XX.append(im)
    X.clear()

    n = int(len(XX) * config.trainSizeScale)
    for i in range(n):
        j = i
        if j > len(XX)-1:
            j %= len(XX)-1
        X.append(XX[j])
    X.setDataInfo(XX)

def dataSplit(*arg):
    if len(arg)==5:
        X, v1, v2, X1, X2 = arg
        if v2 < v1:
            raise Exception('Error')
        X1.clear()
        X2.clear()
        X1.setDataInfo(X)
        X2.setDataInfo(X)
        n1 = int(X.count*v1)
        n2 = int(X.count*v2)
        for i in range(X.count):
            if i>=n1 and i<n2:
                X1.add(X[i])
            else:
                X2.add(X[i])
    elif len(arg)==4:
        X, v, X1, X2 = arg
        X1.clear()
        X2.clear()
        X1.setDataInfo(X)
        X2.setDataInfo(X)
        n = int(X.count*v)
        for i in range(X.count):
            if i<n:
                X1.add(X[i])
            else:
                X2.add(X[i])
    else:
        raise Exception('Error')

def loadDataForCV(config, XList, XXList):
    XList.clear()
    XXList.clear()
    X = dataSet(config.fFeatureTrain, config.fGoldTrain)
    step = 1.0/config.nCV
    i = 0.
    while i<1:
        Xi = dataSet()
        XRest_i = dataSet()
        dataSplit(X, i, i+step, Xi, XRest_i)
        XList.append(XRest_i)
        XXList.append(Xi)
        i+=step
    print('Done! cross-validation train/test data sizes (cv_1, ..., cv_n): ')
    config.swLog.write('Done! cross-validation train/test data sizes (cv_1, ..., cv_n): \n')
    for i in range(config.nCV):
        strlog = '{}/{}, '.format(XList[i].Count, XXList[i].Count)
        print(strlog)
        config.swLog.write(strlog+'\n')

# TODO
# directoryCheck
# readCommand
# helpCommand

def clearDir(d):
    if os.path.isdir(d):
        for f in os.listdir(d):
            clearDir(d+'/'+f)
        #os.removedirs(d)
    else:
        os.remove(d)

if __name__ == '__main__':
    starttime = time.time()
    if len(sys.argv)<4:
        print('Wrong  inputs')
        sys.exit()
    elif not os.path.exists(sys.argv[2]):
        print('file does not exist.')
        sys.exit()
    config = Config()
    config.runMode = sys.argv[1]
    if config.runMode == 'train':
        config.trainFile = sys.argv[2]
        config.testFile = sys.argv[3]
    else:
        config.readFile = sys.argv[2]
        config.outputFile = sys.argv[3]
    if not os.path.exists(config.tempFile):
        os.makedirs(config.tempFile)
    if not os.path.exists(config.tempFile+'/output'):
        os.mkdir(config.tempFile+'/output')
    if not os.path.exists(config.modelDir):
        os.mkdir(config.modelDir)
    run(config)
    clearDir(config.tempFile)
    print('Total time: '+str(time.time()-starttime))

