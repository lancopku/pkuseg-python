import os
import tempfile

class Config:
    lineEnd = '\n'
    biLineEnd = '\n\n'
    triLineEnd = '\n\n\n'
    undrln = '_'
    blank = ' '
    tab = '\t'
    star = '*'
    slash = '/'
    comma = ','
    delimInFeature = '.'
    B = 'B'
    chnNum = '几二三四五六七八九十千万亿兆零'
    engNum = '0123456789.１２３４５６７８９０'
    num = '0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％'
    letter = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－'
    orgAppndx = '院部社国会所办库长委业协宫大局厂业区队帮'
    nsLastWords = '地区市县乡区洲国村'
    ntLastWords = '局院委会协联厂大中行艺站足办总队专所化党法部师校汽电新处室支贸司班垒监革厅小工高社科检系署百盟学旅组警段险团馆庭矿政'
    mark = '*'
    normalLetter = 'abcdefghigklmnopqrstuvwxyz'
        
    def __init__(self):
        # main setting
        self.trainFile = 'data/small_training.utf8'
        self.testFile = 'data/small_test.utf8'
        self.homepath = tempfile.gettempdir()
        self.tempFile = os.path.join(self.homepath, '.pkuseg/temp')
        self.readFile = 'data/small_test.utf8'
        self.outputFile = 'data/small_test_output.utf8'

        self.runMode = "test" # train (normal training), train.rich (training with rich edge features), test, tune£¬ tune.rich, cv (cross validation), cv.rich
        self.modelOptimizer = "crf.adf" # crf.sgd/sgder/adf/bfgs
        self.rate0 = 0.05 # init value of decay rate in SGD and ADF training
        self.regs = [1]
        self.regList = self.regs.copy()
        self.random = 0 # 0 for 0-initialization of model weights, 1 for random init of model weights
        self.evalMetric = "f1" # tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        self.trainSizeScale = 1 # for scaling the size of training data
        self.ttlIter = 20 # of training iterations
        self.nUpdate = 10 # for ADF training
        self.outFolder = self.tempFile + "/" + "output/"
        self.save = 1 # save model file
        self.rawResWrite = True
        self.miniBatch = 1 # mini-batch in stochastic training
        self.nCV = 4 # automatic #-fold cross validation
        self.threadXX = None
        self.nThread = 10 # number of processes
        self.edgeReduce = 0.4
        self.useTraditionalEdge = True
        # ADF training
        self.upper = 0.995 # was tuned for nUpdate = 10
        self.lower = 0.6 # was tuned for nUpdate = 10

        # general
        self.tuneSplit = 0.8 # size of data split for tuning
        self.debug = False # some debug code will run in debug mode
        # SGD training
        self.decayFactor = 0.94 # decay factor in SGD training
        self.scalarResetStep = 1000
        # LBFGS training
        self.mBFGS = 10 # history of 10 iterations of gradients to estimate Hessian info
        self.wolfe = False # for convex & non-convex objective function
        # tuning
        self.iterTuneStoch = 30 # default 30

        # global variables
        self.chunkTagMap = {}
        self.metric = None
        self.ttlScore = 0
        self.interval=None
        self.scoreListList = []
        self.timeList = []
        self.errList = []
        self.diffList = []
        self.reg = 3
        self.glbIter = 0
        self.diff = 1e100 # relative difference from the previous object value, for convergence test
        self.countWithIter = 0
        #self.outDir = ""
        self.outDir = self.outFolder
        self.testrawDir = "rawinputs/"
        self.testinputDir = "inputs/"
        self.tempDir = os.path.join(self.homepath, '.pkuseg/temp')
        self.testoutputDir = "entityoutputs/"

        self.GL_init = True
        self.weightRegMode = "L2" # choosing weight regularizer: L2, L1, GL (groupLasso)
        self.fTrain = self.tempFile + "/" + "train.txt"
        self.fTest = self.tempFile + "/" + "test.txt"
        self.fDev = self.tempFile + "/" + "dev.txt"


        self.dev = False # for testing also on dev data
        self.formatConvert = True


        self.fTune = "tune.txt"
        self.fLog = "trainLog.txt"
        self.fResSum = "summarizeResult.txt"
        self.fResRaw = "rawResult.txt"
        self.fOutput = "outputTag.txt"


        self.fFeatureTrain = self.tempFile + "/" + "ftrain.txt"
        self.fGoldTrain = self.tempFile + "/" + "gtrain.txt"
        self.fFeatureTest = self.tempFile + "/" + "ftest.txt"
        self.fGoldTest = self.tempFile + "/" + "gtest.txt"


        
        self.modelDir = os.path.dirname(os.path.realpath(__file__))+"/models/msra"
        self.fModel = self.modelDir + "/model.txt"

        # feature
        self.numLetterNorm = True
        self.featureTrim = 0
        self.wordFeature = True
        self.wordMax = 6
        self.wordMin = 2
        self.nLabel = 5
        self.order = 1

    def globalCheck(self):
        if self.runMode.find('test')>=0:
            self.ttlIter = 1
        if self.evalMetric == 'f1':
            self.getChunkTagMap()
        if self.evalMetric == 'f1':
            self.metric = 'f-score'
        elif self.evalMetric == 'tok.acc':
            self.metric = 'token-accuracy'
        elif self.evalMetric == 'str.acc':
            self.metric = 'string-accuracy'
        else:
            raise Exception('error')
        assert self.rate0>0
        assert self.trainSizeScale>0
        assert self.ttlIter>0
        assert self.nUpdate>0
        assert self.miniBatch>0
        for reg in self.regList:
            assert reg>=0

    def getChunkTagMap(self):
        self.chunkTagMap={}
        with open(self.modelDir+'/tagIndex.txt', encoding='utf-8') as f:
            a = f.read()
            a = a.replace('\r', '')
            ary = a.split(self.lineEnd)
            for im in ary:
                if im == '':
                    continue
                imAry = im.split(self.blank)
                index = int(imAry[1])
                tagAry = imAry[0].split(self.star)
                tag = tagAry[-1]
                if tag.startswith('I'):
                    tag = 'I'
                if tag.startswith('O'):
                    tag = 'O'
                self.chunkTagMap[index] = tag

    def reinitGlobal(self):
        self.diff = 1e100
        self.countWithIter = 0
        self.glbIter = 0
                    


config = Config()
