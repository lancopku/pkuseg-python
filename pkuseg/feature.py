from multiprocessing import Process, Queue

class Feature:
    def __init__(self, config, file, phase):
        self.config = config
        self.file = file
        self.phase = phase
        if phase == 'train':
            self.train_init()
        else:
            if file is None:
                self.test_wofile()
            else:
                self.test_init()

    def train_init(self):
        config = self.config
        self.trainLexiconSet = set()
        self.trainBigramSet = set()
        self.convertTrain(self.file, config.tempFile+'/c.train.txt', True)
        self.saveBigramFeature(config.modelDir+'/bigram_word.txt')
        self.saveUnigram(config.modelDir+'/unigram_word.txt')
        self.getFeatureSet(config.tempFile+'/c.train.txt')
        self.saveFeature(config.modelDir+'/featureSet.txt')
        self.processFile(config.tempFile+'/c.train.txt', config.tempFile+'/train.txt')

    def test_wofile(self):
        config = self.config
        self.trainLexiconSet = set()
        self.trainBigramSet = set()
        self.readBigramFeature(config.modelDir+'/bigram_word.txt')
        self.readUnigram(config.modelDir+'/unigram_word.txt')
        self.readFeature(config.modelDir+'/featureSet.txt')

    def test_init(self):
        config = self.config
        self.trainLexiconSet = set()
        self.trainBigramSet = set()
        self.readBigramFeature(config.modelDir+'/bigram_word.txt')
        self.readUnigram(config.modelDir+'/unigram_word.txt')
        self.readFeature(config.modelDir+'/featureSet.txt')
        self.convertTest(self.file, config.tempFile+'/c.test.txt', config.tempFile+'/test.raw.txt')
        self.processFile(config.tempFile+'/c.test.txt', config.tempFile+'/test.txt')
        

    def keywordTransfer(self, w):
        if w in '-._,|/*:':
            return '&'
        return w

    def saveUnigram(self, file):
        f = open(file, 'w', encoding='utf-8')
        for w in self.trainLexiconSet:
            f.write(w+'\n')
        f.close()
    def readUnigram(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            self.trainLexiconSet.add(line.strip())
    def saveBigramFeature(self, file):
        f = open(file, 'w', encoding='utf-8')
        for w in self.trainBigramSet:
            f.write(w+'\n')
        f.close()
    def readBigramFeature(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            self.trainBigramSet.add(line.strip())

    def readFeature(self, file):
        featureList = []
        for i in range(10):
            featureList.append([])
            with open(file + "_" + str(i), encoding='utf-8') as sr:
                lines = sr.readlines()
            for line in lines:
                featureList[i].append(line.strip())
        self.featureSet = set()
        for i in range(10):
            for k in range(len(featureList[i])):
                self.featureSet.add(featureList[i][k])
            

    def convertTrain(self, trainfile, outfile, collectInfo):
        config = self.config
        fin = open(trainfile, 'r', encoding='utf-8')
        txt = fin.read()
        fin.close()
        txt = txt.replace('\r', '')
        txt = txt.replace('\t', ' ')
        fout = open(outfile, 'w', encoding='utf-8')
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

        lens = [0]*20
        ary = txt.split(config.lineEnd)
        for im in ary:
            if len(im) == 0:
                continue
            imary = im.split(config.blank)
            tmpary = []
            for w in imary:
                if len(w) != 0:
                    tmpary.append(w)
            imary = tmpary
            for w in imary:
                if collectInfo:
                    self.trainLexiconSet.add(w)
                    if len(w)<=15:
                        lens[len(w)] += 1
                position = 0
                for c in w:
                    if c.strip() == '':
                        continue
                    c = self.keywordTransfer(c)
                    if len(w) == 1:
                        fout.write(c+' '+B_single+'\n')
                    elif position == 0:
                        fout.write(c+' '+B+'\n')
                    elif position==len(w)-1:
                        fout.write(c+' '+I_end+'\n')
                    elif position == 1:
                        fout.write(c+' '+I_first+'\n')
                    else:
                        fout.write(c+' '+I+'\n')
                    position += 1
            fout.write('\n')
            if collectInfo:
                for i in range(1, len(imary)):
                    self.trainBigramSet.add(imary[i-1]+'*'+imary[i])
        
        fout.close()
        if collectInfo:
            for i in range(1,16):
                print('length = %d : %d'%(i, lens[i]))

    def convertTest(self, trainfile, outfile, rawfile):
        config = self.config
        fin = open(trainfile, 'r', encoding='utf-8')
        txt = fin.read()
        fin.close()
        txt = txt.replace('\r', '')
        txt = txt.replace('\t', ' ')
        fout = open(outfile, 'w', encoding='utf-8')
        fraw = open(rawfile, 'w', encoding='utf-8')
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

        ary = txt.split(config.lineEnd)
        for im in ary:
            if len(im) == 0:
                continue
            imary = im.split(config.blank)
            for w in imary: 
                if len(w) == 0:
                    continue
                position = 0
                for c in w:
                    if c.strip() == '':
                        continue
                    fraw.write(c)
                    c = self.keywordTransfer(c)
                    if len(w) == 1:
                        fout.write(c+' '+B_single+'\n')
                    elif position == 0:
                        fout.write(c+' '+B+'\n')
                    elif position==len(w)-1:
                        fout.write(c+' '+I_end+'\n')
                    elif position == 1:
                        fout.write(c+' '+I_first+'\n')
                    else:
                        fout.write(c+' '+I+'\n')
                    position += 1
            fout.write('\n')
            fraw.write('\n')
        fout.close()
        fraw.close()

    def saveFeature(self, file):
        featureList = list(self.featureSet)
        num = len(featureList)//10
        for i in range(10):
            l = i*num
            r = (i+1)*num if i<9 else len(featureList)
            sw = open(file+'_'+str(i), 'w', encoding='utf-8')
            for w in range(l,r):
                word = featureList[w]
                sw.write(word+'\n')
            sw.close()

    def processFile(self, file, file1):
        wordSeqList = []
        tagSeqList = []
        self.normalize(file, wordSeqList, tagSeqList)
        self.writeFeaturesTag(wordSeqList, tagSeqList, file1)

    def writeFeaturesTag(self, wordSeqList, tagSeqList, file):
        config = self.config
        with open(file, 'w', encoding='utf-8') as fout:
            featureList = [None]*len(wordSeqList)
            interval = (len(wordSeqList)+config.nThread-1)//config.nThread
            procs = []
            Q = Queue(5000)
            for i in range(config.nThread):
                start = i*interval
                end = min(start+interval, len(wordSeqList))
                proc = Process(target=Feature.Parallel_FT, args=(self, wordSeqList, tagSeqList, start, end, Q))
                proc.start()
                procs.append(proc)
            for i in range(len(wordSeqList)):
                t = Q.get()
                j, writeFeature = t
                featureList[j] = writeFeature
            for proc in procs:
                proc.join()
            for f in featureList:
                fout.write(f)

    def Parallel_FT(self, wordSeqList, tagSeqList, start, end, Q):
        config = self.config
        for i in range(start, end):
            wordSeq = wordSeqList[i]
            wordAry = wordSeq.split(config.blank)
            tagSeq = tagSeqList[i]
            tagAry = tagSeq.split(config.blank)
            length = len(wordAry)
            writeFeature = []
            for k in range(length):
                nodeFeatures = []
                self.getNodeFeatures(k, wordAry, nodeFeatures)
                writeFeature.append(wordAry[k] + ' ')
                for f in nodeFeatures:
                    if f == '/':
                        writeFeature.append('/ ')
                    else:
                        fAry = f.split(config.slash)
                        id = fAry[0]
                        if id in self.featureSet:
                            writeFeature.append(f+' ')
                        else:
                            writeFeature.append('/ ')
                writeFeature.append(tagAry[k]+'\n')
            writeFeature.append('\n')
            writeFeature = ''.join(writeFeature)
            Q.put((i, writeFeature))
            
                    

    def getFeatureSet(self, file):
        config = self.config
        self.featureSet = set()
        print('getting feature set')
        wordlst = []
        taglst = []
        self.normalize(file, wordlst, taglst)
        featureFreqMap = {}
        for ws in wordlst:
            wordary = ws.split(config.blank)
            for k, w in enumerate(wordary):
                nodeFeatures = []
                self.getNodeFeatures(k, wordary, nodeFeatures)
                for f in nodeFeatures:
                    if f == '/':
                        continue
                    fary = f.split(config.slash)
                    id = fary[0]
                    # id = f ?
                    if not id in featureFreqMap:
                        featureFreqMap[id] = 0
                    featureFreqMap[id] += 1
        for k in featureFreqMap:
            if featureFreqMap[k] > config.featureTrim:
                self.featureSet.add(k)

    def normalize(self, file, wordlst, taglst):
        config = self.config
        with open(file, 'r', encoding='utf-8') as f:
            txt = f.read()
        txt.replace('\t',' ')
        txt.replace('\r', '')
        txt.replace('/', '$')
        ary = txt.split(config.biLineEnd)
        for im in ary:
            if len(im)==0:
                continue
            tmpword = []
            tmptag = []
            imary = im.split(config.lineEnd)
            preTag = config.B
            for imm in imary:
                if len(imm) == 0:
                    continue
                biary = imm.split(config.blank)
                if config.numLetterNorm:
                    tmp1 = biary[0]
                    if tmp1 in config.num:
                        biary[0] = '**Num'
                    if tmp1 in config.letter:
                        biary[0] = '**Letter'
                tmp2 = biary[1]
                if config.order == 2:
                    biary[1] = preTag + config.mark + tmp2
                preTag = tmp2

                tmpword.append(biary[0])
                tmptag.append(biary[1])
            wordlst.append(config.blank.join(tmpword))
            taglst.append(config.blank.join(tmptag))

    def getNodeFeatures(self, n, wordary, flist):
        config = self.config
        w = wordary[n]
        flist.append('$$')
        flist.append('c.'+w)
        if n>0:
            flist.append('c-1.'+wordary[n-1])
        else:
            flist.append('/')
        if n<len(wordary)-1:
            flist.append('c1.'+wordary[n+1])
        else:
            flist.append('/')
        if n>1:
            flist.append('c-2.'+wordary[n-2])
        else:
            flist.append('/')
        if n<len(wordary)-2:
            flist.append('c2.'+wordary[n+2])
        else:
            flist.append('/')
        if n>0:
            flist.append('c-1c.'+wordary[n-1]+config.delimInFeature+w)
        else:
            flist.append('/')
        if n<len(wordary)-1:
            flist.append('cc1.'+w+config.delimInFeature+wordary[n+1])
        else:
            flist.append('/')
        if n>1:
            flist.append('c-2c-1.'+wordary[n-2]+config.delimInFeature+wordary[n-1])
        else:
            flist.append('/')

        # no num/letter based features

        if config.wordFeature:
            tmplst = []
            for l in range(config.wordMax, config.wordMin-1, -1):
                tmp = getCharSeq(wordary, n-l+1, l)
                if tmp != '':
                    if tmp in self.trainLexiconSet:
                        flist.append('w-1.'+tmp)
                        tmplst.append(tmp)
                    else:
                        flist.append('/')
                        tmplst.append('**noWord')
                else:
                    flist.append('/')
                    tmplst.append('**noWord')
            prelst_in = tmplst

            tmplst = []
            for l in range(config.wordMax, config.wordMin-1, -1):
                tmp = getCharSeq(wordary, n, l)
                if tmp != '':
                    if tmp in self.trainLexiconSet:
                        flist.append('w1.'+tmp)
                        tmplst.append(tmp)
                    else:
                        flist.append('/')
                        tmplst.append('**noWord')
                else:
                    flist.append('/')
                    tmplst.append('**noWord')
            postlst_in = tmplst

            tmplst = []
            for l in range(config.wordMax, config.wordMin-1, -1):
                tmp = getCharSeq(wordary, n-l, l)
                if tmp != '':
                    if tmp in self.trainLexiconSet:
                        tmplst.append(tmp)
                    else:
                        tmplst.append('**noWord')
                else:
                    tmplst.append('**noWord')
            prelst_ex = tmplst

            tmplst = []
            for l in range(config.wordMax, config.wordMin-1, -1):
                tmp = getCharSeq(wordary, n+1, l)
                if tmp != '':
                    if tmp in self.trainLexiconSet:
                        tmplst.append(tmp)
                    else:
                        tmplst.append('**noWord')
                else:
                    tmplst.append('**noWord')
            postlst_ex = tmplst

            for pre in prelst_ex:
                for post in postlst_in:
                    bigram = pre + '*' + post
                    if bigram in self.trainBigramSet:
                        flist.append('ww.l.'+bigram)
                    else:
                        flist.append('/')

            for pre in prelst_in:
                for post in postlst_ex:
                    bigram = pre + '*' + post
                    if bigram in self.trainBigramSet:
                        flist.append('ww.r.'+bigram)
                    else:
                        flist.append('/')

def getCharSeq(sen, i, length):
    if i<0 or i>=len(sen):
        return ''
    if i+length-1>=len(sen):
        return ''
    return ''.join(sen[i:i+length])

                
if __name__ == '__main__':
    f = Feature('train', 'data/small_training.utf8')



