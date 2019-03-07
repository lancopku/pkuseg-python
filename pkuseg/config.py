import os
import tempfile


class Config:
    lineEnd = "\n"
    biLineEnd = "\n\n"
    triLineEnd = "\n\n\n"
    undrln = "_"
    blank = " "
    tab = "\t"
    star = "*"
    slash = "/"
    comma = ","
    delimInFeature = "."
    B = "B"
    num = "0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％"
    letter = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－"
    mark = "*"
    model_urls = {
        "postag": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
        "medicine": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/medicine.zip",
        "tourism": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/tourism.zip",
        "news": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/news.zip",
        "web": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/web.zip",
    }
    model_hash = {
        "postag": "afdf15f4e39bc47a39be4c37e3761b0c8f6ad1783f3cd3aff52984aebc0a1da9",
        "medicine": "773d655713acd27dd1ea9f97d91349cc1b6aa2fc5b158cd742dc924e6f239dfc",
        "tourism": "1c84a0366fe6fda73eda93e2f31fd399923b2f5df2818603f426a200b05cbce9",
        "news": "18188b68e76b06fc437ec91edf8883a537fe25fa606641534f6f004d2f9a2e42",
        "web": "4867f5817f187246889f4db259298c3fcee07c0b03a2d09444155b28c366579e",
    }
    available_models = ["default", "medicine", "tourism", "web", "news"]
    models_with_dict = ["medicine", "tourism"]


    def __init__(self):
        # main setting
        self.pkuseg_home = os.path.expanduser(os.getenv('PKUSEG_HOME', '~/.pkuseg'))
        self.trainFile = os.path.join("data", "small_training.utf8")
        self.testFile = os.path.join("data", "small_test.utf8")
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.homepath = self._tmp_dir.name
        self.tempFile = os.path.join(self.homepath, ".pkuseg", "temp")
        self.readFile = os.path.join("data", "small_test.utf8")
        self.outputFile = os.path.join("data", "small_test_output.utf8")

        self.modelOptimizer = "crf.adf"
        self.rate0 = 0.05  # init value of decay rate in SGD and ADF training
        # self.reg = 1
        # self.regs = [1]
        # self.regList = self.regs.copy()
        self.random = (
            0
        )  # 0 for 0-initialization of model weights, 1 for random init of model weights
        self.evalMetric = (
            "f1"
        )  # tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        self.trainSizeScale = 1  # for scaling the size of training data
        self.ttlIter = 20  # of training iterations
        self.nUpdate = 10  # for ADF training
        self.outFolder = os.path.join(self.tempFile, "output")
        self.save = 1  # save model file
        self.rawResWrite = True
        self.miniBatch = 1  # mini-batch in stochastic training
        self.nThread = 10  # number of processes
        # ADF training
        self.upper = 0.995  # was tuned for nUpdate = 10
        self.lower = 0.6  # was tuned for nUpdate = 10

        # global variables
        self.metric = None
        self.reg = 1
        self.outDir = self.outFolder
        self.testrawDir = "rawinputs/"
        self.testinputDir = "inputs/"
        self.tempDir = os.path.join(self.homepath, ".pkuseg", "temp")
        self.testoutputDir = "entityoutputs/"

        # self.GL_init = True
        self.weightRegMode = "L2"  # choosing weight regularizer: L2, L1)

        self.c_train = os.path.join(self.tempFile, "train.conll.txt")
        self.f_train = os.path.join(self.tempFile, "train.feat.txt")

        self.c_test = os.path.join(self.tempFile, "test.conll.txt")
        self.f_test = os.path.join(self.tempFile, "test.feat.txt")

        self.fTune = "tune.txt"
        self.fLog = "trainLog.txt"
        self.fResSum = "summarizeResult.txt"
        self.fResRaw = "rawResult.txt"
        self.fOutput = "outputTag-{}.txt"

        self.fFeatureTrain = os.path.join(self.tempFile, "ftrain.txt")
        self.fGoldTrain = os.path.join(self.tempFile, "gtrain.txt")
        self.fFeatureTest = os.path.join(self.tempFile, "ftest.txt")
        self.fGoldTest = os.path.join(self.tempFile, "gtest.txt")

        self.modelDir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "models", "ctb8"
        )
        self.fModel = os.path.join(self.modelDir, "model.txt")

        # feature
        self.numLetterNorm = True
        self.featureTrim = 0
        self.wordFeature = True
        self.wordMax = 6
        self.wordMin = 2
        self.nLabel = 5
        self.order = 1

    def globalCheck(self):
        if self.evalMetric == "f1":
            self.metric = "f-score"
        elif self.evalMetric == "tok.acc":
            self.metric = "token-accuracy"
        elif self.evalMetric == "str.acc":
            self.metric = "string-accuracy"
        else:
            raise Exception("invalid eval metric")
        assert self.rate0 > 0
        assert self.trainSizeScale > 0
        assert self.ttlIter > 0
        assert self.nUpdate > 0
        assert self.miniBatch > 0
        assert self.reg > 0


config = Config()
