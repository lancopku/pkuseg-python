import os
import numpy as np
from .config import Config

class model:
    def __init__(self, config, *args):
        if len(args)==1:
            file = args[0]
            if os.path.exists(file):
                self.load(file)
            else:
                print(file)
        elif len(args)==2:
            if type(args[1])==bool:
                m, wCopy = args
                self.NTag = m.NTag
                if wCopy:
                    self.W = m.W.copy()
                else:
                    self.W = np.zeros_like(m.W)
            else:
                X, fGen = args
                self.NTag = X.NTag
                if config.random == 0:
                    self.W = np.zeros(fGen.NCompleteFeature)
                elif config.random == 1:
                    self.W = np.random.random(size=(fGen.NCompleteFeature,))*2-1
                else:
                    raise Exception('error')
        else:
            raise Exception('error')
        
    def load(self, file):
        with open(file, encoding='utf-8') as f:
            txt = f.read()
        txt = txt.replace('\r', '')
        ary = txt.split(Config.lineEnd)
        self.NTag = int(ary[0])
        wsize = int(ary[1])
        self.W = np.zeros(wsize)
        for i in range(2, wsize):
            self.W[i-2] = float(ary[i])

    def save(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            f.write(str(self.NTag)+'\n')
            f.write(str(self.W.shape[0])+'\n')
            for im in self.W:
                f.write('%.4f\n'%im)




            
