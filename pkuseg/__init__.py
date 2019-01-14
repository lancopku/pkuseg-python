from .feature import *
from .config import Config
from .data_format import *
from .toolbox import *
from .inference import *
from .model import *
from .main import *
import os
from multiprocessing import Process, Queue


class TrieNode:
    """建立词典的Trie树节点"""

    def __init__(self, isword):
        self.isword = isword
        self.children = {}


class Preprocesser:
    """预处理器，在用户词典中的词强制分割"""

    def __init__(self, dict_file):
        """初始化建立Trie树"""
        self.dict_data = dict_file
        if isinstance(dict_file, str):
            with open(dict_file, encoding="utf-8") as f:
                lines = f.readlines()
            self.trie = TrieNode(False)
            for line in lines:
                self.insert(line.strip())
        else:
            self.trie = TrieNode(False)
            for w in dict_file:
                assert isinstance(w, str)
                self.insert(w.strip())

    def insert(self, word):
        """Trie树中插入单词"""
        l = len(word)
        now = self.trie
        for i in range(l):
            c = word[i]
            if not c in now.children:
                now.children[c] = TrieNode(False)
            now = now.children[c]
        now.isword = True

    def solve(self, txt):
        """对文本进行预处理"""
        outlst = []
        iswlst = []
        l = len(txt)
        last = 0
        i = 0
        while i < l:
            now = self.trie
            j = i
            found = False
            while True:
                c = txt[j]
                if not c in now.children:
                    break
                now = now.children[c]
                j += 1
                if now.isword:
                    found = True
                    break
                if j == l:
                    break
            if found:
                if last != i:
                    outlst.append(txt[last:i])
                    iswlst.append(False)
                outlst.append(txt[i:j])
                iswlst.append(True)
                last = j
                i = j
            else:
                i += 1
        if last < l:
            outlst.append(txt[last:l])
            iswlst.append(False)
        return outlst, iswlst


class pkuseg:
    def __init__(self, model_name="ctb8", user_dict=[]):
        """初始化函数，加载模型及用户词典"""
        print("loading model")
        config = Config()
        self.config = config
        if model_name in ["ctb8"]:
            config.modelDir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "models", model_name
            )
        else:
            config.modelDir = model_name
        config.fModel = os.path.join(config.modelDir, "model.txt")
        if user_dict == "safe_lexicon":
            file_name = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "dicts/safe_lexicon.txt"
            )
        else:
            file_name = user_dict
        self.preprocesser = Preprocesser(file_name)
        self.testFeature = Feature(
            config, None, "test"
        )  # convert: keywordtransfor, process: test.txt
        self.df = dataFormat(config)  # test.txt -> ftest.txt
        self.df.readFeature(config.modelDir + "/featureIndex.txt")
        self.df.readTag(config.modelDir + "/tagIndex.txt")
        self.model = model(config, config.fModel)
        self.idx2tag = [None] * len(self.df.tagIndexMap)
        for i in self.df.tagIndexMap:
            self.idx2tag[self.df.tagIndexMap[i]] = i
        if config.nLabel == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.nLabel == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 5:
            B = "B"
            B_single = "B_single"
            I_first = "I_first"
            I = "I"
            I_end = "I_end"
        self.B = B
        self.B_single = B_single
        self.I_first = I_first
        self.I = I
        self.I_end = I_end
        print("finish")

    def cut(self, txt):
        """分词，结果返回一个list"""
        config = self.config
        txt = txt.strip()
        # txt = txt.replace("\r", "")
        txt = txt.replace("\t", " ")
        ary = txt.split(config.lineEnd)
        ret = []
        for im in ary:
            if len(im) == 0:
                continue
            imary = im.split(config.blank)
            tmpary = []
            for w0 in imary:
                if len(w0) == 0:
                    tmpary.append(w0)
                    continue
                lst, isword = self.preprocesser.solve(w0)
                for w, isw in zip(lst, isword):
                    if isw:
                        ret.append(w)
                        continue
                    transed = []
                    for i, c in enumerate(w):
                        x = self.testFeature.keywordTransfer(c)
                        if config.numLetterNorm:
                            if x in config.num:
                                x = "**Num"
                            if x in config.letter:
                                x = "**Letter"
                        transed.append(x)
                    l = len(transed)
                    all_features = []
                    for i in range(l):
                        nodeFeatures = []
                        self.testFeature.getNodeFeatures(i, transed, nodeFeatures)
                        for j, f in enumerate(nodeFeatures):
                            if f != "/":
                                id = f.split(config.slash)[0]  # id == f?
                                if not id in self.testFeature.featureSet:
                                    nodeFeatures[j] = "/"
                        all_features.append(nodeFeatures)
                    all_feature_idx = []
                    for k in range(l):
                        flag = 0
                        ary = all_features[k]
                        featureLine = []
                        for i, f in enumerate(ary):
                            if f == "/":
                                continue
                            ary2 = f.split(config.slash)
                            tmp = []
                            for j in ary2:
                                if j != "":
                                    tmp.append(j)
                            ary2 = tmp
                            feature = str(i + 1) + "." + ary2[0]
                            value = ""
                            real = False
                            if len(ary2) > 1:
                                value = ary2[1]
                                real = True
                            if not feature in self.df.featureIndexMap:
                                continue
                            flag = 1
                            fIndex = self.df.featureIndexMap[feature]
                            if not real:
                                featureLine.append(str(fIndex))
                            else:
                                featureLine.append(str(fIndex) + "/" + value)
                        if flag == 0:
                            featureLine.append("0")
                        all_feature_idx.append(featureLine)
                    XX = dataSet()
                    XX.nFeature = len(self.df.featureIndexMap)
                    XX.nTag = len(self.df.tagIndexMap)
                    seq = dataSeq()
                    seq.load(all_feature_idx)
                    XX.append(seq)
                    tb = toolbox(config, XX, False, self.model)
                    taglist = tb.test(XX, 0, dynamic=True)
                    taglist = taglist[0].split(",")[:-1]
                    out = []
                    now = ""
                    isstart = True
                    for t, k in zip(taglist, w):
                        if isstart:
                            now = k
                            isstart = False
                        elif self.idx2tag[int(t)].find("B") >= 0:
                            out.append(now)
                            now = k
                        else:
                            now = now + k
                    out.append(now)
                    ret.extend(out)
        return ret


def train(trainFile, testFile, savedir, nthread=10):
    """用于训练模型"""
    config = Config()
    starttime = time.time()
    if not os.path.exists(trainFile):
        raise Exception("trainfile does not exist.")
    if not os.path.exists(testFile):
        raise Exception("testfile does not exist.")
    if not os.path.exists(config.tempFile):
        os.makedirs(config.tempFile)
    if not os.path.exists(config.tempFile + "/output"):
        os.mkdir(config.tempFile + "/output")
    config.runMode = "train"
    config.trainFile = trainFile
    config.testFile = testFile
    config.modelDir = savedir
    config.fModel = os.path.join(config.modelDir, "model.txt")
    config.nThread = nthread
    run(config)
    clearDir(config.tempFile)
    print("Total time: " + str(time.time() - starttime))


def _test_single_proc(
    input_file, output_file, model_name="ctb8", user_dict=None, verbose=False
):

    times = []
    times.append(time.time())
    if user_dict is None:
        user_dict = []
    seg = pkuseg(model_name, user_dict)

    times.append(time.time())
    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    times.append(time.time())
    results = []
    for line in lines:
        results.append(" ".join(seg.cut(line)))

    times.append(time.time())
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    times.append(time.time())

    print("total_time:\t{:.3f}".format(times[-1] - times[0]))

    if verbose:
        time_strs = ["load_model", "read_file", "word_seg", "write_file"]
        for key, value in zip(
            time_strs,
            [end - start for start, end in zip(times[:-1], times[1:])],
        ):
            print("{}:\t{:.3f}".format(key, value))


def _proc_deprecated(seg, lines, start, end, q):
    for i in range(start, end):
        l = lines[i].strip()
        ret = seg.cut(l)
        q.put((i, " ".join(ret)))


def _proc(seg, in_queue, out_queue):
    # TODO: load seg (json or pickle serialization) in sub_process
    #       to avoid pickle seg online when using start method other
    #       than fork
    while True:
        item = in_queue.get()
        if item is None:
            return
        idx, line = item
        out_queue.put((idx, " ".join(seg.cut(line))))


def _test_multi_proc(
    input_file,
    output_file,
    nthread,
    model_name="ctb8",
    user_dict=None,
    verbose=False,
):

    times = []
    times.append(time.time())
    if user_dict is None:
        user_dict = []
    seg = pkuseg(model_name, user_dict)

    times.append(time.time())
    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    times.append(time.time())
    in_queue = Queue()
    out_queue = Queue()
    procs = []
    for i in range(nthread):
        p = Process(target=_proc, args=(seg, in_queue, out_queue))
        procs.append(p)

    for idx, line in enumerate(lines):
        in_queue.put((idx, line))

    for proc in procs:
        in_queue.put(None)
        proc.start()

    times.append(time.time())
    result = [None] * len(lines)
    for _ in result:
        idx, line = out_queue.get()
        result[idx] = line

    times.append(time.time())
    for p in procs:
        p.join()

    times.append(time.time())
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result))
    times.append(time.time())

    print("total_time:\t{:.3f}".format(times[-1] - times[0]))

    if verbose:
        time_strs = [
            "load_model",
            "read_file",
            "start_proc",
            "word_seg",
            "join_proc",
            "write_file",
        ]

        for key, value in zip(
            time_strs,
            [end - start for start, end in zip(times[:-1], times[1:])],
        ):
            print("{}:\t{:.3f}".format(key, value))


def test(
    input_file,
    output_file,
    model_name="ctb8",
    user_dict=None,
    nthread=10,
    verbose=False,
):

    if nthread > 1:
        _test_multi_proc(
            input_file, output_file, nthread, model_name, user_dict, verbose
        )
    else:
        _test_single_proc(
            input_file, output_file, model_name, user_dict, verbose
        )


