import numpy as np
from .viterbi import *

class belief:
    def __init__(self, nNodes, nStates):
        self.belState = np.zeros((nNodes, nStates))
        self.belEdge = [None]
        for i in range(1, nNodes):
            self.belEdge.append(np.zeros((nStates, nStates)))
        self.Z = 0

class inference:
    def __init__(self, tb):
        self._optim = tb.Optim
        self._fGene = tb.FGene
        self._grad = tb.Grad
    def getYYandY(self, m, x, YYlist, Ylist, maskYYlist, maskYlist):
        nNodes = len(x)
        nTag = m.NTag
        dAry = np.zeros(nTag)
        mask = False
        # TODO try except?
        #try:
        for i in range(nNodes):
            YYi = np.zeros((nTag, nTag))
            Yi = dAry.copy()
            self.getLogYY(m, x, i, YYi, Yi, False, mask)
            YYlist.append(YYi)
            Ylist.append(Yi)
            maskYYlist.append(YYi.copy())
            maskYlist.append(Yi.copy())
        #except:
        #    print('read out time!')
        maskValue = -1e100
        for i in range(nNodes):
            Y = maskYlist[i]
            tagList = x.getTags()
            for s in range(len(Y)):
                if tagList[i]!=s:
                    Y[s] = maskValue

    def getBeliefs(self, bel, m, x, YYlist, Ylist):
        if type(Ylist)==bool:
            self.getBeliefs_scalar(bel, m, x, YYlist, Ylist)
            return
        nNodes = len(x)
        nTag = m.NTag
        dAry = np.zeros(nTag)
        alpha_Y = dAry.copy()
        newAlpha_Y = dAry.copy()
        for i in range(nNodes-1, 0, -1):
            YY = YYlist[i].copy()
            Y = Ylist[i]
            tmp_Y = bel.belState[i] + Y
            bel.belState[i-1] = logMultiply(YY, tmp_Y)
        for i in range(nNodes):
            YY = None
            if i>0:
                YY = YYlist[i].copy()
            Y = Ylist[i]
            if i>0:
                tmp_Y = alpha_Y.copy()
                YY = YY.transpose()
                newAlpha_Y = logMultiply(YY, tmp_Y)
                newAlpha_Y = newAlpha_Y + Y
            else:
                newAlpha_Y = Y.copy()
            if i>0:
                tmp_Y = Y + bel.belState[i]
                YY = YY.transpose()
                bel.belEdge[i] = YY
                for yPre in range(nTag):
                    for y in range(nTag):
                        bel.belEdge[i][yPre, y] += tmp_Y[y] + alpha_Y[yPre]
            bel.belState[i] = bel.belState[i] + newAlpha_Y
            alpha_Y = newAlpha_Y
        Z = logSum(alpha_Y)
        for i in range(nNodes):
            bel.belState[i] = np.exp(bel.belState[i] - Z)
        for i in range(1, nNodes):
            bel.belEdge[i] = np.exp(bel.belEdge[i]-Z)
        bel.Z = Z

    def getBeliefs_scalar(self, bel, m, x, scalar, mask):
        nNodes = len(x)
        nTag = m.NTag
        YY = np.zeros((nTag, nTag))
        dAry = np.zeros(nTag)
        Y = dAry.copy()
        alpha_Y = dAry.copy()
        newAlpha_Y = dAry.copy()
        tmp_Y = dAry.copy()
        for i in range(nNodes-1, 0, -1):
            getLogYY(m, x, i, YY, Y, False, mask, scalar)
            tmp_Y = bel.belState[i] + Y
            bel.belState[i-1] = logMultiply(YY, tmp_Y)
        for i in range(nNodes):
            getLogYY(m, x, i, YY, Y, False, mask, scalar)
            if i>0:
                YY = YYlist[i].copy()
            Y = Ylist[i]
            if i>0:
                tmp_Y = alpha_Y.copy()
                YY = YY.transpose()
                newAlpha_Y = logMultiply(YY, tmp_Y)
                newAlpha_Y = newAlpha_Y + Y
            else:
                newAlpha_Y = Y.copy()
            if i>0:
                tmp_Y = Y + bel.belState[i]
                YY = YY.transpose()
                bel.belEdge[i] = YY
                for yPre in range(nTag):
                    for y in range(nTag):
                        bel.belEdge[i][yPre, y] += tmp_Y[y] + alpha_Y[yPre]
            alpha_Y = newAlpha_Y
        Z = logSum(alpha_Y)
        for i in range(1, nNodes):
            bel.belEdge[i] = np.exp(bel.belEdge[i]-Z)
        bel.Z = Z

    def getLogYY(self, m, x, i, YY, Y, takeExp, mask, scalar=1.):
        YY.fill(1)
        Y.fill(0)
        w = m.W
        fList = self._fGene.getFeatureTemp(x, i)
        nTag = m.NTag
        for ft in fList:
            for s in range(nTag):
                f = self._fGene.getNodeFeatID(ft.id, s)
                Y[s] += w[f]*scalar*ft.val
        if i>0:
            for s in range(nTag):
                for sPre in range(nTag):
                    f = self._fGene.getEdgeFeatID(sPre, s)
                    YY[sPre, s]+=w[f] * scalar
        maskValue = -1e100
        if takeExp:
            Y = np.exp(Y)
            YY = np.exp(YY)
            maskValue = 0
        if mask:
            tagList = x.getTags()
            for s in range(Y.shape[0]):
                if tagList[i] != s:
                    Y[s] = maskValue

    def decodeViterbi_fast(self, m, x, tags):
        tags.clear()
        nNode = len(x)
        nTag = m.NTag
        YY = np.zeros((nTag, nTag))
        Y = np.zeros(nTag)
        viter = Viterbi(nNode, nTag)
        for i in range(nNode):
            self.getLogYY(m, x, i, YY, Y, False, False)
            viter.setScores(i, Y, YY)
        numer = viter.runViterbi(tags, False)
        return numer
        

def logMultiply(A, B):
    toSumLists = np.zeros_like(A)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            toSumLists[r,c] = A[r,c]+B[c]
    ret = np.zeros(A.shape[0])
    for r in range(A.shape[0]):
        ret[r] = logSum(toSumLists[r])
    return ret

def logSum(a):
    s = a[0]
    for i in range(1, len(a)):
        if s >=a[i]:
            m1, m2 = s, a[i]
        else:
            m1, m2 = a[i], s
        s = m1 + np.log(1+np.exp(m2-m1))
    return s
                
        
