from multiprocessing import Process, Queue
from .featuregenerator import *
from .model import *
from .inference import *
from .gradient import *
from .optimizer import *
from .dataformat import *
import time

class toolbox:
    def __init__(self, config, *args):
        self.config = config
        if len(args)==2:
            if args[1]:
                self.X = args[0]
                self.FGene = featureGenerator(config, self.X)
                self.Model = model(config, self.X, self.FGene)
                self.Optim = None
                self.Grad = None
                self.Inf = inference(self)
                self.Grad = gradient(self)
                self.initOptimizer()
            else:
                self.X = args[0]
                self.FGene = featureGenerator(config, self.X)
                self.Model = model(config, config.fModel)
                self.Optim = None
                self.Grad = None
                self.Inf = inference(self)
                self.Grad = gradient(self)
        elif len(args)==3:
            self.X = args[0]
            self.FGene = featureGenerator(config, self.X)
            self.Model = args[2]
            self.Optim = None
            self.Grad = None
            self.Inf = inference(self)
            self.Grad = gradient(self)
        else:
            raise Exception('Unknown toolbox args')
            
    def initOptimizer(self):
        config = self.config
        if config.modelOptimizer.startswith('crf'):
            if config.modelOptimizer.endswith('sgd') or config.modelOptimizer.endswith('sgder') or config.modelOptimizer.endswith('adf'):
                self.Optim = optimStochastic(config, self)
            elif config.modelOptimizer.endswith('bfgs'):
                self.Optim = optimBFGS(self, self.Model.W, config.mBFGS, 0, config.ttlIter)
            else:
                raise Exception('error.')
        else:
            raise Exception('error.')

    def train(self):
        return self.Optim.optimize()

    def test(self, X, a1, dynamic=False):
        config = self.config
        if type(a1) == str:
            outfile = a1
            config.swOutput = open(outfile, 'w')
            if config.evalMetric == 'tok.acc':
                scoreList = decode_tokAcc(X, self.Model, dynamic)
            elif config.evalMetric == 'str.acc':
                scoreList = decode_strAcc(X, self.Model, dynamic)
                decode_tokAcc(X, self.Model) #this is only for record accuracy info on trainLog, this is useful in t-test 
            elif config.evalMetric == 'f1':
                scoreList = decode_fscore(X, self.Model, dynamic)
                decode_tokAcc(X, self.Model)
            else:
                raise Exception('error.')
            config.swOutput.close()
            return scoreList
        else:
            it = a1
            outfile = config.outDir + config.fOutput
            if not dynamic:
                config.swOutput = open(outfile, 'w')
            if config.evalMetric == 'tok.acc':
                scoreList = self.decode_tokAcc(X, self.Model, dynamic)
            elif config.evalMetric == 'str.acc':
                scoreList = self.decode_strAcc(X, self.Model, dynamic)
            elif config.evalMetric == 'f1':
                scoreList = self.decode_fscore(X, self.Model, dynamic)
            else:
                raise Exception('error.')
            if not dynamic:
                config.swOutput.close()
            return scoreList

    # token accuracy
    def decode_tokAcc(self, X, m, dynamic=False):
        config = self.config
        nTag = m.NTag
        tmpAry = [0]*nTag
        corrOutput = [0]*nTag
        gold = [0]*nTag
        output = [0]*nTag
        X2 = []
        multiThreading(X, X2, dynamic)
        for x in X2:
            outTags = x._yOutput
            goldTags = x._x.getTags()
            if config.swOutput is not None:
                for i in range(len(outTags)):
                    config.swOutput.write(str(outTags[i])+',')
                config.swOutput.write('\n')
            for i in range(len(outTags)):
                gold[goldTags[i]]+=1
                output[outTags[i]]+=1
                if outTags[i] == goldTags[i]:
                    corrOutput[outTags[i]]+=1
        config.swLog.write('% tag-type  #gold  #output  #correct-output  token-precision  token-recall  token-f-score\n')
        sumGold = 0
        sumOutput = 0
        sumCorrOutput = 0
        for i in range(nTag):
            sumCorrOutput += corrOutput[i]
            sumGold += gold[i]
            sumOutput += output[i]
            if gold[i]==0:
                rec = 0
            else:
                rec = corrOutput[i]*100.0/gold[i]
            if output[i]==0:
                prec = 0
            else:
                prec = corrOutput[i]*100.0/output[i]
            config.swLog.write('% {}:  {}  {}  {}  {}  {}  {}\n'.format(i, gold[i], output[i], corrOutput[i], '%.2f'%prec, '%.2f'%rec, '%.2f'%(2 * prec * rec / (prec + rec))))
        if sumGold == 0:
            rec = 0
        else:
            rec = sumCorrOutput*100.0/sumGold
        if sumOutput == 0:
            prec = 0
        else:
            prec = sumCorrOutput*100.0/sumOutput

        if prec==0 and rec==0:
            fscore=0
        else:
            fscore = 2*prec*rec/(prec+rec)
        config.swLog.write('% overall-tags:  {}  {}  {}  {}  {}  {}\n'.format(sumGold, sumOutput, sumCorrOutput, '%.2f'%prec, '%.2f'%rec, '%.2f'%fscore))
        config.swLog.flush()
        return [fscore]

    def decode_strAcc(self, X, m, dynamic=False):
        config = self.config
        xsize = len(X)
        corr = 0
        X2 = []
        multiThreading(X, X2, dynamic)
        for x in X2:
            if config.swOutput is not None:
                for i in range(len(x._x)):
                    config.swOutput.write(x._yOutput[i]+',')
                config.swOutput.write('\n')
            goldTags = x._x.getTags()
            ck = True
            for i in range(len(x._x)):
                if goldTags[i] != x._yOutput[i]:
                    ck = False
                    break
            if ck:
                corr += 1
        acc = corr / xsize * 100.0
        config.swLog.write('total-tag-strings={}  correct-tag-strings={}  string-accuracy={}%'.format(xsize, corr, acc))
        return [acc]

    def decode_fscore(self, X, m, dynamic=False):
        config = self.config
        X2 = []

        self.multiThreading(X, X2, dynamic)
        goldTagList = []
        resTagList = []
        for x in X2:
            res = ''
            for im in x._yOutput:
                res += str(im)+','
            resTagList.append(res)
            if not dynamic:
                if config.swOutput is not None:
                    for i in range(len(x._yOutput)):
                        config.swOutput.write(str(x._yOutput[i])+',')
                    config.swOutput.write('\n')
            goldTags = x._x.getTags()
            gold = ''
            for im in goldTags:
                gold += str(im)+','
            goldTagList.append(gold)
        if dynamic:
            return resTagList
        scoreList = []
        if config.runMode == 'train':
            infoList = []
            scoreList = getFscore(config, goldTagList, resTagList, infoList)
            config.swLog.write('#gold-chunk={}  #output-chunk={}  #correct-output-chunk={}  precision={}  recall={}  f-score={}\n'.format(infoList[0], infoList[1], infoList[2], '%.2f'%scoreList[1], '%.2f'%scoreList[2], '%.2f'%scoreList[0]))
        return scoreList

    def multiThreading(self, X, X2, dynamic=False):
        config = self.config
        if dynamic:
            for i in range(len(X)):
                X2.append(dataSeqTest(X[i], []))
            for k, x in enumerate(X2):
                tags = []
                prob = self.Inf.decodeViterbi_fast(self.Model, x._x, tags)
                X2[k]._yOutput.clear()
                X2[k]._yOutput.extend(tags)
            return
        for i in range(len(X)):
            X2.append(dataSeqTest(X[i], []))
        if len(X)<config.nThread:
            config.nThread = len(X)
        interval = (len(X2)+config.nThread-1)//config.nThread
        procs = []
        Q = Queue(5000)
        for i in range(config.nThread):
            start = i*interval
            end = min(start+interval, len(X2))
            proc = Process(target=toolbox.taskRunner_test, args=(self, X2, start, end, Q))
            proc.start()
            procs.append(proc)
        for i in range(len(X2)):
            t = Q.get()
            k, tags = t
            X2[k]._yOutput.clear()
            X2[k]._yOutput.extend(tags)
        for proc in procs:
            proc.join()


    def taskRunner_test(self, X2, start, end, Q):
        for k in range(start, end):
            x = X2[k]
            tags = []
            prob = self.Inf.decodeViterbi_fast(self.Model, x._x, tags)
            Q.put((k, tags))



def getFscore(config, goldTagList, resTagList, infoList):
    scoreList = []
    assert len(resTagList) == len(goldTagList)
    getNewTagList(config.chunkTagMap, goldTagList)
    getNewTagList(config.chunkTagMap, resTagList)
    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0
    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(config.comma)
        tmp = []
        for t in resChunkAry:
            if len(t)>0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(config.comma)
        tmp = []
        for t in goldChunkAry:
            if len(t)>0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1
    pre = correct_chunk/res_chunk*100
    rec = correct_chunk/gold_chunk*100
    f1 = 0 if correct_chunk == 0 else 2*pre*rec/(pre+rec)
    scoreList.append(f1)
    scoreList.append(pre)
    scoreList.append(rec)
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    return scoreList

def getNewTagList(tagMap, tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        for i in range(len(tagAry)):
            if tagAry[i]=='':
                continue
            index = int(tagAry[i])
            if not index in tagMap:
                raise Exception('error')
            tagAry[i] = tagMap[index]
        newTags = ','.join(tagAry)
        tmpList.append(newTags)
    tagList.clear()
    for im in tmpList:
        tagList.append(im)

def getChunks(tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        tmp = []
        for t in tagAry:
            if t != '':
                tmp.append(t)
        tagAry = tmp
        chunks = ''
        for i in range(len(tagAry)):
            if tagAry[i].startswith('B'):
                pos = i
                length = 1
                ty = tagAry[i]
                for j in range(i+1, len(tagAry)):
                    if tagAry[j] == 'I':
                        length += 1
                    else:
                        break
                chunk = ty + '*' + str(length) + '*' + str(pos)
                chunks = chunks + chunk + ','
        tmpList.append(chunks)
    return tmpList


    
