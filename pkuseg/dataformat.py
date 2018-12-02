from .config import Config
from .featuregenerator import *
import os
import copy
import random

class dataFormat:
    def __init__(self, config):
        self.featureIndexMap = {}
        self.tagIndexMap = {}
        self.config = config
    def convert(self):
        config = self.config
        if config.runMode.find('train')>=0:
            self.getMaps(config.fTrain)
            self.saveFeature(config.modelDir+'/featureIndex.txt')
            self.convertFile(config.fTrain)
        else:
            self.readFeature(config.modelDir+'/featureIndex.txt')
            self.readTag(config.modelDir+'/tagIndex.txt')
        self.convertFile(config.fTest)
        if config.dev:
            self.convertFile(config.fDev)

    def saveFeature(self, file):
        featureList = list(self.featureIndexMap.keys())
        num = len(featureList)//10
        for i in range(10):
            l = i*num
            r = (i+1)*num if i<9 else len(featureList)
            sw = open(file+'_'+str(i), 'w', encoding='utf-8')
            for w in range(l,r):
                word = featureList[w]
                sw.write(word+' '+str(self.featureIndexMap[word])+'\n')
            sw.close()

    def readFeature(self, file):
        featureList = []
        for i in range(10):
            featureList.append([])
            with open(file+'_'+str(i), encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                featureList[i].append(line.strip())
        feature = []
        for i in range(10):
            for line in featureList[i]:
                word, index = line.split(' ')
                self.featureIndexMap[word] = int(index)

    def readFeatureNormal(self, path):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            u, v = line.split(' ')
            self.featureIndexMap[u] = int(v)

    def readTag(self, path):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            u, v = line.split(' ')
            self.tagIndexMap[u] = int(v)

    def getMaps(self, file):
        config = self.config
        if not os.path.exists(file):
            print('file {} not exist!'.format(file))
        print('file {} converting...'.format(file))
        featureFreqMap = {}
        tagSet = set()
        with open(file, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\t', ' ')
            line = line.replace('\r', '').strip()
            if line == '':
                continue
            ary = line.split(config.blank)
            for i in range(1, len(ary)-1):
                if ary[i] == '' or ary[i]=='/':
                    continue
                if config.weightRegMode == 'GL':
                    if not config.GL_init and config.groupTrim[i-1]:
                        continue

                ary2 = ary[i].split(config.slash)
                feature = str(i)+'.'+ary2[0]
                if not feature in featureFreqMap:
                    featureFreqMap[feature] = 0
                featureFreqMap[feature]+=1
            tag = ary[-1]
            tagSet.add(tag)
        sortList = []
        for k in featureFreqMap:
            sortList.append(k+' '+str(featureFreqMap[k]))
        if config.weightRegMode == 'GL':
            sortList.sort(key=lambda x:(int(x.split(config.blank)[1].strip()), x))
            with open('featureTemp_sorted.txt', 'w', encoding = 'utf-8') as f:
                for x in sortList:
                    f.write(x+'\n')
            config.groupStart = [0]
            config.groupEnd = []
            for k in range(1, len(sortList)):
                thisAry = sortList[k].split(config.dot)
                preAry = sortList[k-1].split(config.dot)
                s = thisAry[0]
                preAry = preAry[0]
                if s != preAry:
                    config.groupStart.append(k)
                    config.groupEnd.append(k)
            config.groupEnd.append(len(sortList))
        else:
            sortList.sort(key=lambda x:(int(x.split(config.blank)[1].strip()), x), reverse=True)

        if config.weightRegMode == 'GL' and config.GL_init:
            if nFeatTemp != len(config.groupStart):
                raise Exception("inconsistent # of features per line, check the feature file for consistency!")
        with open(config.modelDir + '/featureIndex.txt', 'w', encoding = 'utf-8') as swFeat:
            for i, l in enumerate(sortList):
                ary = l.split(config.blank)
                self.featureIndexMap[ary[0]] = i
                swFeat.write('{} {}\n'.format(ary[0].strip(), i))
        with open(config.modelDir + '/tagIndex.txt', 'w', encoding = 'utf-8') as swTag:
            tagSortList = []
            for tag in tagSet:
                tagSortList.append(tag)
            tagSortList.sort()
            for i, l in enumerate(tagSortList):
                self.tagIndexMap[l] = i
                swTag.write('{} {}\n'.format(l, i))

    def convertFile(self, file):
        config = self.config
        if not os.path.exists(file):
            print('file {} not exist!'.format(file))
        print('file converting...')
        if file==config.fTrain:
            swFeature = open(config.fFeatureTrain, 'w', encoding='utf-8')
            swGold = open(config.fGoldTrain, 'w', encoding='utf-8')
        else:
            swFeature = open(config.fFeatureTest, 'w', encoding='utf-8')
            swGold = open(config.fGoldTest, 'w', encoding='utf-8')
        swFeature.write(str(len(self.featureIndexMap))+'\n\n')
        swGold.write(str(len(self.tagIndexMap))+'\n\n')
        with open(file, encoding = 'utf-8') as sr:
            readLines = sr.readlines()
        featureList = []
        goldList = []
        for k in range(len(readLines)):
            line = readLines[k]
            line = line.replace('\t', '').strip()
            featureLine = ''
            goldLine = ''
            if line == '':
                featureLine = featureLine+'\n'
                goldLine = goldLine+'\n\n'
                featureList.append(featureLine)
                goldList.append(goldLine)
                continue
            flag = 0
            ary = line.split(config.blank)
            tmp = []
            for i in ary:
                if i != '':
                    tmp.append(i)
            ary = tmp
            for i in range(1, len(ary)-1):
                if ary[i] == '/':
                    continue
                ary2 = ary[i].split(config.slash)
                tmp = []
                for j in ary2:
                    if j != '':
                        tmp.append(j)
                ary2 = tmp
                feature = str(i)+'.'+ary2[0]
                value = ''
                real = False
                if len(ary2)>1:
                    value = ary2[1]
                    real = True
                if not feature in self.featureIndexMap:
                    continue
                flag = 1
                fIndex = self.featureIndexMap[feature]
                if not real:
                    featureLine = featureLine + str(fIndex) + ','
                else:
                    featureLine = featureLine + str(fIndex) + '/' + value + ','
            if flag == 0:
                featureLine = featureLine + '0'
            featureLine = featureLine + '\n'
            tag = ary[-1]
            tIndex = self.tagIndexMap[tag]
            goldLine = goldLine + str(tIndex)+','
            featureList.append(featureLine)
            goldList.append(goldLine)
        for i in range(len(featureList)):
            swFeature.write(featureList[i])
            swGold.write(goldList[i])
        swFeature.close()
        swGold.close()

class dataSet:
    def __init__(self, *args):
        self.lst = []
        self.nTag = 0
        self.nFeature = 0
        if len(args)==2:
            if type(args[0]) == int:
                self.nTag, self.nFeature = args
            else:
                self.load(args[0], args[1])
    def __len__(self):
        return len(self.lst)
    def __iter__(self):
        return self.iterator()
    def __getitem__(self, x):
        return self.lst[x]
    def iterator(self):
        for i in self.lst:
            yield i
    def append(self,x):
        self.lst.append(x)
    def clear(self):
        self.lst = []
    def randomShuffle(self):
        cp = copy.deepcopy(self)
        random.shuffle(cp.lst)
        return cp
    def setDataInfo(self, X):
        self.nTag = X.nTag
        self.nFeature = X.nFeature
    def load(self, fileFeature, fileTag):
        srfileFeature = open(fileFeature, encoding='utf-8')
        srfileTag = open(fileTag, encoding='utf-8')
        txt = srfileFeature.read()
        txt.replace('\r', '')
        fAry = txt.split(Config.biLineEnd)
        tmp = []
        for i in fAry:
            if i != '':
                tmp.append(i)
        fAry = tmp
        txt = srfileTag.read()
        txt.replace('\r', '')
        tAry = txt.split(Config.biLineEnd)
        tmp = []
        for i in tAry:
            if i != '':
                tmp.append(i)
        tAry = tmp

        assert len(fAry) == len(tAry)
        self.nFeature = int(fAry[0])
        self.nTag = int(tAry[0])
        for i in range(1, len(fAry)):
            features = fAry[i]
            tags = tAry[i]
            seq = dataSeq()
            seq.read(features, tags)
            self.append(seq)
        srfileFeature.close()
        srfileTag.close()
    @property
    def NTag(self):
        return self.nTag

class dataSeq:
    def __init__(self, *args):
        self.featureTemps = []
        self.yGold = []
        if len(args)==2:
            self.featureTemps = copy.deepcopy(args[0])
            self.yGold = copy.deepcopy(args[1])
        elif len(args)==3:
            x,n,length = args
            end=min(n+length, len(x))
            for i in range(n, end):
                self.featureTemps.append(x.featureTemps[i])
                yGold.append(x.yGold[i])
    def __len__(self):
        return len(self.featureTemps)
    def read(self, a, b):
        lineAry = a.split(Config.lineEnd)
        for im in lineAry:
            if im == '':
                continue
            nodeList = []
            imAry = im.split(Config.comma)
            for imm in imAry:
                if imm == '':
                    continue
                if imm.find('/')>=0:
                    biAry = imm.split(Config.slash)
                    ft = featureTemp(int(biAry[0], float(biAry[1])))
                    nodeList.append(ft)
                else:
                    ft = featureTemp(int(imm), 1)
                    nodeList.append(ft)
            self.featureTemps.append(nodeList)
        lineAry = b.split(Config.comma)
        for im in lineAry:
            if im == '':
                continue
            self.yGold.append(int(im))
    def load(self, feature):
        for imAry in feature:
            nodeList = []
            for imm in imAry:
                if imm == '':
                    continue
                if imm.find('/')>=0:
                    biAry = imm.split(Config.slash)
                    ft = featureTemp(int(biAry[0], float(biAry[1])))
                    nodeList.append(ft)
                else:
                    ft = featureTemp(int(imm), 1)
                    nodeList.append(ft)
            self.featureTemps.append(nodeList)
            self.yGold.append(0)
    def getFeatureTemp(self, *args):
        return self.featureTemps if len(args)==0 else self.featureTemps[args[0]]
    def getTags(self, *args):
        return self.yGold if len(args)==0 else self.yGold[args[0]]
    def setTags(self, lst):
        assert len(lst)==len(self.yGold)
        for i in range(len(lst)):
            self.yGold[i]=lst[i]
            
class dataSeqTest:
    def __init__(self, x, yOutput):
        self._x = x
        self._yOutput = yOutput
        
            



            
