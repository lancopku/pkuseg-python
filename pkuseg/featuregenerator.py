class featureTemp:
    def __init__(self, a, b):
        self.id = a
        self.val = b

class featureGenerator:
    def __init__(self, config, *args):
        self._nFeatureTemp = 0
        self._nCompleteFeature = 0
        self._backoff1 = 0
        self._nTag = 0
        if len(args)==1:
            X = args[0]
            self._nFeatureTemp = X.nFeature
            self._nTag = X.nTag
            try:
                config.swLog.write('feature templates: {}\n'.format(self._nFeatureTemp))
            except:
                pass
            nNodeFeature = self._nFeatureTemp * self._nTag
            nEdgeFeature = self._nTag * self._nTag
            self._backoff1 = nNodeFeature
            self._nCompleteFeature = nNodeFeature + nEdgeFeature
            try:
                config.swLog.write('complete features: {}\n'.format(self._nCompleteFeature))
            except:
                pass
    def getFeatureTemp(self, x, node):
        return x.getFeatureTemp(node)
    def getNodeFeatID(self, id, s):
        return id*self._nTag+s
    def getEdgeFeatID(self, sPre, s):
        return self._backoff1 + s*self._nTag + sPre
    @property
    def NCompleteFeature(self):
        return self._nCompleteFeature
    @property
    def Backoff1(self):
        return self._backoff1
    
