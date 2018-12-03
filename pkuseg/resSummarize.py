import numpy as np
from .config import Config


def tomatrix(s):
    lines = s.split(Config.lineEnd)
    lst = []
    for line in lines:
        if line == '':
            continue
        if not line.startswith('%'):
            tmp = []
            for i in line.split(Config.comma):
                tmp.append(float(i))
            lst.append(tmp)
    return np.array(lst)

def summarize(config, fn='f2'):
    with open(config.outDir+config.fResRaw, encoding='utf-8') as sr:
        txt = sr.read()
    tst = txt.replace('\r', '')
    regions = txt.split(config.triLineEnd)
    with open(config.outDir + config.fResSum, 'w', encoding='utf-8') as sw:
        for region in regions:
            if region == '':
                continue
            blocks = region.split(config.biLineEnd)
            mList = []
            for im in blocks:
                mList.append(tomatrix(im))
            avgM = np.zeros_like(mList[0])
            for m in mList:
                avgM = avgM + m
            avgM = avgM / len(mList)
            sqravgM = np.zeros_like(mList[0])
            for m in mList:
                sqravgM += m*m
            sqravgM = sqravgM / len(mList)
            deviM = (sqravgM-avgM*avgM)**0.5
            sw.write('%averaged values:\n')
            for i in range(avgM.shape[0]):
                for j in range(avgM.shape[1]):
                    sw.write(('%.2f'%avgM[i,j])+',')
                sw.write('\n')
            sw.write('\n%deviations:\n')
            for i in range(deviM.shape[0]):
                for j in range(deviM.shape[1]):
                    sw.write(('%.2f'%deviM[i,j])+',')
                sw.write('\n')
            sw.write('\n%avg & devi:\n')
            for i in range(avgM.shape[0]):
                for j in range(avgM.shape[1]):
                    sw.write(('%.2f'%avgM[i,j])+'+-'+('%.2f'%deviM[i,j])+',')
                sw.write('\n')
            sw.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n')

def write(config):
    if config.runMode.find('test')>=0:
        assert len(config.scoreListList)<=1 and len(config.timeList)<=1
        if config.rawResWrite:
            config.swResRaw.write('%test results:\n')
        lst = config.scoreListList[0]
        if config.evalMetric == 'f1':
            if config.rawResWrite:
                config.swResRaw.write('% f-score={}%  precision={}%  recall={}%  '.format('%.2f'%lst[0], '%.2f'%lst[1], '%.2f'%lst[2]))
        else:
            if config.rawResWrite:
                config.swResRaw.write('% {}={}%  '.format(config.metric, '%.2f'%lst[0]))
        for im in lst:
            if config.rawResWrite:
                config.swResRaw.write(('%.2f'%im)+',')
    else:
        if config.rawResWrite:
            config.swResRaw.write('% training results:'+config.metric+'\n')
        for i in range(config.ttlIter):
            it = i+1
            if config.rawResWrite:
                config.swResRaw.write('% iter#={}  '.format(it))
            lst = config.scoreListList[i]
            if config.evalMetric == 'f1':
                if config.rawResWrite:
                    config.swResRaw.write('% f-score={}%  precision={}%  recall={}%  '.format('%.2f'%lst[0], '%.2f'%lst[1], '%.2f'%lst[2]))
            else:
                if config.rawResWrite:
                    config.swResRaw.write('% {}={}%  '.format(config.metric, '%.2f'%lst[0]))
            time = 0
            for k in range(i+1):
                time += config.timeList[k]
            if config.rawResWrite:
                config.swResRaw.write('cumulative-time(sec)={}  objective={}  diff={}\n'.format('%.2f'%time, '%.2f'%config.errList[i], '%.2f'%config.diffList[i]))
        config.ttlScore = 0
        for i in range(config.ttlIter):
            it = i+1
            if config.rawResWrite:
                config.swResRaw.write('% iter#={}  '.format(it))
            lst = config.scoreListList[i]
            config.ttlScore += lst[0]
            if config.evalMetric == 'f1':
                if config.rawResWrite:
                    config.swResRaw.write('% f-score={}%  precision={}%  recall={}%  '.format('%.2f'%lst[0], '%.2f'%lst[1], '%.2f'%lst[2]))
            else:
                if config.rawResWrite:
                    config.swResRaw.write('% {}={}%  '.format(config.metric, '%.2f'%lst[0]))
            time = 0
            for k in range(i+1):
                time += config.timeList[k]
            if config.rawResWrite:
                config.swResRaw.write('cumulative-time(sec)={}  objective={}  diff={}\n'.format('%.2f'%time, '%.2f'%config.errList[i], '%.2f'%config.diffList[i]))
        config.scoreListList.clear()
        config.timeList.clear()
        config.errList.clear()
        config.diffList.clear()
            
                    
