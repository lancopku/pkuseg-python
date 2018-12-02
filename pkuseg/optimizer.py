import numpy as np
import random

class optimizer:
    def __init__(self):
        self._preVals = []
    def optimize(self):
        raise Exception('error')
    def convergeTest(self, err):
        val = 1e100
        if len(self._preVals)>1:
            prevVal = self._preVals[0]
            if len(self._preVals)==10:
                self._preVals.pop(0)
            avgImprovement = (prevVal-err)/len(self._preVals)
            relAvg = avgImprovement/abs(err)
            val = relAvg
        self._preVals.append(err)
        return val

class optimStochastic(optimizer):
    def __init__(self, config, tb):
        self.config = config
        optimizer.__init__(self)
        self._model = tb.Model
        self._X = tb.X
        self._inf = tb.Inf
        self._fGene = tb.FGene
        self._grad = tb.Grad
        config.decayList = np.ones_like(self._model.W)*config.rate0

    def optimize(self):
        config = self.config
        error = 0
        if config.modelOptimizer.endswith('adf'):
            error = self.adf()
        elif config.modelOptimizer.endswith('sgder'):
            error = self.sgd_exactReg()
        else:
            error = self.sgd_lazyReg()
        config.swLog.flush()
        return error

    def adf(self):
        config = self.config
        w = self._model.W
        fsize = w.shape[0]
        xsize = len(self._X)
        grad = np.zeros(fsize)
        error = 0
        featureCountList = [0] * fsize
        ri = list(range(xsize))
        random.shuffle(ri)
        config.interval = xsize // config.nUpdate
        nSample = 0
        for t in range(0,xsize,config.miniBatch):
            XX = []
            end=False
            for k in range(t, t+config.miniBatch):
                i = ri[k]
                x = self._X[i]
                XX.append(x)
                if k==xsize-1:
                    end=True
                    break
            mbSize = len(XX)
            nSample += mbSize
            fSet = set()
            err = self._grad.getGrad_SGD_miniBatch(grad, self._model, XX, fSet)
            error += err
            for i in fSet:
                featureCountList[i] += 1
            check=False
            for k in range(t, t+config.miniBatch):
                if t != 0 and k%config.interval == 0:
                    check=True
            # update decay rates
            if check or end:
                for i in range(fsize):
                    v = featureCountList[i]
                    u = v/nSample
                    eta = config.upper - (config.upper-config.lower)*u
                    config.decayList[i] *= eta
                for i in range(len(featureCountList)):
                    featureCountList[i] = 0
            # update weights
            for i in fSet:
                w[i] -= config.decayList[i] * grad[i]
                grad[i] = 0
            # reg
            if check or end:
                if config.reg != 0:
                    for i in range(fsize):
                        grad_i = w[i]/(config.reg*config.reg)*(nSample/xsize)
                        w[i] -= config.decayList[i] * grad_i
                nSample = 0
            config.countWithIter += mbSize
        if config.reg != 0:
            s = (w*w).sum()
            error += s/(2.0*config.reg*config.reg)
        config.diff = self.convergeTest(error)
        return error
    def sgd_lazyReg(self):
        raise Exception('Not implemented')
    def sgd_exactReg(self):
        raise Exception('Not implemented')







