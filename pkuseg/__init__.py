from .feature import *
from .config import config
from .dataformat import *
from .toolbox import *
from .inference import *
from .model import *
from .main import *

class pkuseg:
    def __init__(self):
        print('loading model')
        self.testFeature = Feature(None, 'test')    # convert: keywordtransfor, process: test.txt
        self.df = dataFormat()  # test.txt -> ftest.txt
        self.df.readFeature(config.modelDir+'/featureIndex.txt')
        self.df.readTag(config.modelDir+'/tagIndex.txt')
        self.model = model(config.fModel)
        self.idx2tag = [None]*len(self.df.tagIndexMap)
        for i in self.df.tagIndexMap:
            self.idx2tag[self.df.tagIndexMap[i]] = i
        if config.nLabel == 2:
            B = B_single = 'B'
            I_first = I = I_end = 'I'
        elif config.nLabel == 3:
            B = B_single = 'B'
            I_first = I = 'I'
            I_end = 'I_end'
        elif config.nLabel == 4:
            B = 'B'
            B_single = 'B_single'
            I_first = I = 'I'
            I_end = 'I_end'
        elif config.nLabel == 5:
            B = 'B'
            B_single = 'B_single'
            I_first = 'I_first'
            I = 'I'
            I_end = 'I_end'
        self.B = B
        self.B_single = B_single
        self.I_first = I_first
        self.I = I
        self.I_end = I_end
        print('finish')

    def cut(self, txt):
        txt = txt.replace('\r', '')
        txt = txt.replace('\t', ' ')
        ary = txt.split(config.lineEnd)
        ret = []
        for im in ary:
            if len(im)==0:
                continue
            imary = im.split(config.blank)
            tmpary = []
            for w in imary:
                if len(w) == 0:
                    tmpary.append(w)
                    continue
                transed = []
                for i, c in enumerate(w):
                    x = self.testFeature.keywordTransfer(c)
                    if config.numLetterNorm:
                        if x in config.num:
                            x = '**Num'
                        if x in config.letter:
                            x = '**Letter'
                    transed.append(x)
                l = len(transed)
                all_features = []
                for i in range(l):
                    nodeFeatures = []
                    self.testFeature.getNodeFeatures(i, transed, nodeFeatures)
                    for j, f in enumerate(nodeFeatures):
                        if f != '/':
                            id = f.split(config.slash)[0]   # id == f?
                            if not id in self.testFeature.featureSet:
                                nodeFeatures[j] = '/'
                    all_features.append(nodeFeatures)
                all_feature_idx = []
                for k in range(l):
                    flag = 0
                    ary = all_features[k]
                    featureLine = []
                    for i, f in enumerate(ary):
                        if f == '/':
                            continue
                        ary2 = f.split(config.slash)
                        tmp = []
                        for j in ary2:
                            if j != '':
                                tmp.append(j)
                        ary2 = tmp
                        feature = str(i+1)+'.'+ary2[0]
                        value = ''
                        real = False
                        if len(ary2)>1:
                            value = ary2[1]
                            real = True
                        if not feature in self.df.featureIndexMap:
                            continue
                        flag = 1
                        fIndex = self.df.featureIndexMap[feature]
                        if not real:
                            featureLine.append(str(fIndex))
                        else:
                            featureLine.append(str(fIndex)+'/'+value)
                    if flag == 0:
                        featureLine.append('0')
                    all_feature_idx.append(featureLine)
                XX = dataSet()
                XX.nFeature = len(self.df.featureIndexMap)
                XX.nTag = len(self.df.tagIndexMap)
                seq = dataSeq()
                seq.load(all_feature_idx)
                XX.append(seq)
                tb = toolbox(XX, False, self.model)
                taglist = tb.test(XX, 0, dynamic=True)
                taglist = taglist[0].split(',')[:-1]
                out = []
                now = ''
                for t, k in zip(taglist, w):
                    if self.idx2tag[int(t)].find('B')>=0:
                        out.append(now)
                        now = k
                    else:
                        now = now+k
                out.append(now)
                out = out[1:]
                ret.extend(out)
        return ret

# TODO train
'''
def train(trainFile, testFile):
    starttime = time.time()
    if not os.path.exists(trainFile):
        raise Exception('file does not exist.')
    config.runMode = 'train'
    config.trainFile = trainFile
    config.testFile = testFile
    run()
    clearDir(config.tempFile)
    print('Total time: '+str(time.time()-starttime))
'''

def test(readFile, outputFile):
    starttime = time.time()
    if not os.path.exists(readFile):
        raise Exception('file does not exist.')
    config.runMode = 'test'
    config.readFile = readFile
    config.outputFile = outputFile
    if not os.path.exists(config.tempFile):
        os.makedirs(config.tempFile)
    if not os.path.exists(config.tempFile+'/output'):
        os.mkdir(config.tempFile+'/output')
    if not os.path.exists(config.modelDir):
        os.mkdir(config.modelDir)
    run()
    clearDir(config.tempFile)
    print('Total time: '+str(time.time()-starttime))

