from __future__ import print_function
import sys

if sys.version_info[0] < 3:
    print("pkuseg does not support python2", file=sys.stderr)
    sys.exit(1)

import os
import time
import pickle as pkl
import multiprocessing

from multiprocessing import Process, Queue

import pkuseg.trainer as trainer
import pkuseg.inference as _inf

from pkuseg.config import config
from pkuseg.feature_extractor import FeatureExtractor
from pkuseg.model import Model
from pkuseg.download import download_model
from pkuseg.postag import Postag

model_urls = {
    "postag": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
    "medical": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/medical.zip",
    "tourism": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/tourism.zip",
}

class TrieNode:
    """建立词典的Trie树节点"""

    def __init__(self, isword):
        self.isword = isword
        self.children = {}


class Preprocesser:
    """预处理器，在用户词典中的词强制分割"""

    def __init__(self, dict_file):
        """初始化建立Trie树"""
        if dict_file is None:
            dict_file = []
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

class Postprocesser:
    """对分词结果后处理"""
    def __init__(self, common_name, other_names):
        if common_name is None and other_names is None:
            self.do_process = False
            return
        self.do_process = True
        if common_name is None:
            self.common_words = set()
        else:
            # with open(common_name, encoding='utf-8') as f:
            #     lines = f.readlines()
            # self.common_words = set(map(lambda x:x.strip(), lines))
            with open(common_name, "rb") as f:
                all_words = pkl.load(f).strip().split("\n")
            self.common_words = set(all_words)
        if other_names is None:
            self.other_words = set()
        else:
            self.other_words = set()
            for other_name in other_names:
                # with open(other_name, encoding='utf-8') as f:
                #     lines = f.readlines()
                # self.other_words.update(set(map(lambda x:x.strip(), lines)))
                with open(other_name, "rb") as f:
                    all_words = pkl.load(f).strip().split("\n")
                self.other_words.update(set(all_words))

    def post_process(self, sent, check_seperated):
        for m in reversed(range(2, 8)): 
            end = len(sent)-m
            if end < 0:
                continue
            i = 0
            while (i < end + 1):
                merged_words = ''.join(sent[i:i+m])
                if merged_words in self.common_words:
                    do_seg = True
                elif merged_words in self.other_words:
                    if check_seperated:
                        seperated = all(((w in self.common_words) 
                            or (w in self.other_words)) for w in sent[i:i+m])
                    else:
                        seperated = False
                    if seperated:
                        do_seg = False
                    else:
                        do_seg = True
                else:
                    do_seg = False
                if do_seg:
                    for k in range(m):
                        del sent[i]
                    sent.insert(i, merged_words)
                    i += 1
                    end = len(sent) - m
                else:
                    i += 1 
        return sent

    def __call__(self, sent):
        if not self.do_process:
            return sent
        return self.post_process(sent, check_seperated=True)

class pkuseg:
    def __init__(self, model_name="default", user_dict="default", seg_only=True):
        """初始化函数，加载模型及用户词典"""
        # print("loading model")
        # config = Config()
        # self.config = config
        self.seg_only = seg_only
        if model_name in ["default"]:
            config.modelDir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "models",
                model_name,
            )
        elif model_name in ["medical", "tourism"]:
            config.modelDir = os.path.join(
                config.pkuseg_home,
                model_name,
            )
            download_model(model_urls[model_name], config.pkuseg_home)
        else:
            config.modelDir = model_name
        # config.fModel = os.path.join(config.modelDir, "model.txt")
        if user_dict is None:
            file_name = None
            other_names = None
        else:
            if user_dict != "default":
                file_name = user_dict
            else:
                file_name = None
            if model_name in ["medical", "tourism"]:
                file_name = None
                other_name = os.path.join(
                    config.pkuseg_home,
                    model_name,
                    model_name+"_dict.pkl",
                )
                default_name = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "dicts", "default.pkl",
                )
                other_names = [other_name, default_name]
            else:
                default_name = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "dicts", "default.pkl",
                )
                other_names = [default_name]

        self.preprocesser = Preprocesser(file_name)
        # self.preprocesser = Preprocesser([])
        self.postprocesser = Postprocesser(None, other_names)

        self.feature_extractor = FeatureExtractor.load()
        self.model = Model.load()

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()
        }

        self.n_feature = len(self.feature_extractor.feature_to_idx)
        self.n_tag = len(self.feature_extractor.tag_to_idx)

        if not seg_only:
            download_model(model_urls["postag"], config.pkuseg_home)
            postag_dir = os.path.join(
                config.pkuseg_home,
                "postag",
            )
            self.tagger = Postag(postag_dir)

        # print("finish")

    def _cut(self, text):
        """
        直接对文本分词
        """

        examples = list(self.feature_extractor.normalize_text(text))
        length = len(examples)

        all_feature = []  # type: List[List[int]]
        for idx in range(length):
            node_feature_idx = self.feature_extractor.get_node_features_idx(
                idx, examples
            )
            # node_feature = self.feature_extractor.get_node_features(
            #     idx, examples
            # )

            # node_feature_idx = []
            # for feature in node_feature:
            #     feature_idx = self.feature_extractor.feature_to_idx.get(feature)
            #     if feature_idx is not None:
            #         node_feature_idx.append(feature_idx)
            # if not node_feature_idx:
            #     node_feature_idx.append(0)

            all_feature.append(node_feature_idx)

        _, tags = _inf.decodeViterbi_fast(all_feature, self.model)

        words = []
        current_word = None
        is_start = True
        for tag, char in zip(tags, text):
            if is_start:
                current_word = char
                is_start = False
            elif "B" in self.idx_to_tag[tag]:
                words.append(current_word)
                current_word = char
            else:
                current_word += char
        if current_word:
            words.append(current_word)

        return words

    def cut(self, txt):
        """分词，结果返回一个list"""

        txt = txt.strip()

        ret = []

        if not txt:
            return ret

        imary = txt.split()  # 根据空格分为多个片段

        # 对每个片段分词
        for w0 in imary:
            if not w0:
                continue

            # 根据用户词典拆成更多片段
            lst, isword = self.preprocesser.solve(w0)

            for w, isw in zip(lst, isword):
                if isw:
                    ret.append(w)
                    continue

                output = self._cut(w)
                ret.extend(self.postprocesser(output))
        
        if not self.seg_only:
            tags = self.tagger.tag(ret)
            ret = list(zip(ret, tags))
        return ret


def train(trainFile, testFile, savedir, train_iter=20, init_model=None):
    """用于训练模型"""
    # config = Config()
    starttime = time.time()
    if not os.path.exists(trainFile):
        raise Exception("trainfile does not exist.")
    if not os.path.exists(testFile):
        raise Exception("testfile does not exist.")
    if not os.path.exists(config.tempFile):
        os.makedirs(config.tempFile)
    if not os.path.exists(config.tempFile + "/output"):
        os.mkdir(config.tempFile + "/output")
    # config.runMode = "train"
    config.trainFile = trainFile
    config.testFile = testFile
    config.modelDir = savedir
    # config.fModel = os.path.join(config.modelDir, "model.txt")
    config.nThread = 1
    config.ttlIter = train_iter
    config.init_model = init_model

    os.makedirs(config.modelDir, exist_ok=True)

    trainer.train(config)

    # pkuseg.main.run(config)
    # clearDir(config.tempFile)
    print("Total time: " + str(time.time() - starttime))


def _test_single_proc(
    input_file, output_file, model_name="default", user_dict="default", verbose=False
):

    times = []
    times.append(time.time())
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


def _proc_alt(model_name, user_dict, in_queue, out_queue):
    seg = pkuseg(model_name, user_dict)
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
    model_name="default",
    user_dict="default",
    verbose=False,
):

    alt = multiprocessing.get_start_method() == "spawn"

    times = []
    times.append(time.time())

    if alt:
        seg = None
    else:
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
    for _ in range(nthread):
        if alt:
            p = Process(
                target=_proc_alt,
                args=(model_name, user_dict, in_queue, out_queue),
            )
        else:
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

        if alt:
            times = times[1:]
            time_strs = time_strs[1:]
            time_strs[2] = "load_modal & word_seg"

        for key, value in zip(
            time_strs,
            [end - start for start, end in zip(times[:-1], times[1:])],
        ):
            print("{}:\t{:.3f}".format(key, value))


def test(
    input_file,
    output_file,
    model_name="default",
    user_dict="default",
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

