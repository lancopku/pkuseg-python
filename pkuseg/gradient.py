from .inference import *

class gradient:
    def __init__(self, tb):
        self._optim = tb.Optim
        self._inf = tb.Inf
        self._fGene = tb.FGene
    def getGrad_SGD_miniBatch(self, g, m, X, idset):
        if idset is not None:
            idset.clear()
        error = 0
        for x in X:
            idset2 = set()
            error += self.getGradCRF(g,m,x,idset2)
            if idset is not None:
                for i in idset2:
                    idset.add(i)
        return error
    def getGradCRF(self, *args):
        if len(args) == 4:
            vecGrad, m, x, idSet = args
            sca_version = False
        elif len(args) == 5:
            vecGrad, scalar, m, x, idSet = args
            sca_version = True
        else:
            raise Exception('error.')
        if idSet is not None:
            idSet.clear()
        nTag = m.NTag
        bel = belief(len(x), nTag)
        belMasked = belief(len(x), nTag)
        if not sca_version:
            YYlist = []
            maskYYlist = []
            Ylist = []
            maskYlist = []
            self._inf.getYYandY(m, x, YYlist, Ylist, maskYYlist, maskYlist)
            self._inf.getBeliefs(bel, m, x, YYlist, Ylist)
            self._inf.getBeliefs(belMasked, m, x, maskYYlist, maskYlist)
        else:
            self._inf.getBeliefs(bel, m, x, scalar, False)
            self._inf.getBeliefs(belMasked, m, x, scalar, True)
        ZGold = belMasked.Z
        Z = bel.Z
        for i in range(len(x)):
            fList = self._fGene.getFeatureTemp(x,i)
            for im in fList:
                for s in range(nTag):
                    f = self._fGene.getNodeFeatID(im.id, s)
                    if idSet is not None:
                        idSet.add(f)
                    vecGrad[f] += bel.belState[i][s] * im.val
                    vecGrad[f] -= belMasked.belState[i][s] * im.val
        for i in range(1, len(x)):
            for s in range(nTag):
                for sPre in range(nTag):
                    f = self._fGene.getEdgeFeatID(sPre, s)
                    if idSet is not None:
                        idSet.add(f)
                    vecGrad[f] += bel.belEdge[i][sPre, s]
                    vecGrad[f] -= belMasked.belEdge[i][sPre, s]
        return Z-ZGold
    def getGrad_BFGS(self, *args):
        raise Exception('Not implemented')
    def getGrad_SGD(self, *args):
        raise Exception('Not implemented')
