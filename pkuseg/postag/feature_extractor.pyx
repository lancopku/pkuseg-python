# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3
import json
import os
import sys
import pickle
from collections import Counter
from itertools import product

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_slice_str(iterable, int start, int length, int all_len):
    if start < 0 or start >= all_len:
        return ""
    if start + length >= all_len + 1:
        return ""
    return "".join(iterable[start : start + length])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def __get_node_features_idx(int idx, list nodes not None, dict feature_to_idx not None):

    cdef:
        list flist = []
        Py_ssize_t i = idx
        int length = len(nodes)
        int j


    w = nodes[i]

    # $$ starts feature
    flist.append(0)

    # unigram/bgiram feature
    feat  = "w." + w
    if feat in feature_to_idx:
        feature = feature_to_idx[feat]
        flist.append(feature)

    for j in range(1, 4):
        if len(w)>=j:
            feat = "tr1.pre.%d.%s"%(j, w[:j])
            if feat in feature_to_idx:
                flist.append(feature_to_idx[feat])
            feat = "tr1.post.%d.%s"%(j, w[-j:])
            if feat in feature_to_idx:
                flist.append(feature_to_idx[feat])

    if i > 0:
        feat = "tr1.w-1." + nodes[i - 1]
    else:
        feat = "tr1.w-1.BOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])
    if i < length - 1:
        feat = "tr1.w1." + nodes[i + 1]
    else:
        feat = "tr1.w1.EOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])
    if i > 1:
        feat = "tr1.w-2." + nodes[i - 2]
    else:
        feat = "tr1.w-2.BOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])
    if i < length - 2:
        feat = "tr1.w2." + nodes[i + 2]
    else:
        feat = "tr1.w2.EOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])
    if i > 0:
        feat = "tr1.w_-1_0." + nodes[i - 1] + "." + w
    else:
        feat = "tr1.w_-1_0.BOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])
    if i < length - 1:
        feat = "tr1.w_0_1." + w + "." + nodes[i + 1]
    else:
        feat = "tr1.w_0_1.EOS"
    if feat in feature_to_idx:
        flist.append(feature_to_idx[feat])

    return flist


class FeatureExtractor:

    keywords = "-._,|/*:"

    num = set("0123456789." "几二三四五六七八九十千万亿兆零" "１２３４５６７８９０％")
    letter = set(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ" "ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" "／・－"
    )

    keywords_translate_table = str.maketrans("-._,|/*:", "&&&&&&&&")

    @classmethod
    def keyword_rename(cls, text):
        return text.translate(cls.keywords_translate_table)

    @classmethod
    def _num_letter_normalize(cls, word):
        if not list(filter(lambda x:x not in cls.num, word)):
            return "**Num"
        return word

    @classmethod
    def normalize_text(cls, text):
        for i in range(len(text)):
            text[i] = cls.keyword_rename(text[i])
        for character in text:
            yield cls._num_letter_normalize(character)


    def __init__(self):

        # self.unigram = set()  # type: Set[str]
        # self.bigram = set()  # type: Set[str]
        self.feature_to_idx = {}  # type: Dict[str, int]
        self.tag_to_idx = {}  # type: Dict[str, int]
    
    def get_node_features_idx(self, idx, nodes):
        return __get_node_features_idx(idx, nodes, self.feature_to_idx)

    def get_node_features(self, idx, wordary):
        cdef int length = len(wordary)
        w = wordary[idx]
        flist = []

        # 1 start feature
        flist.append("$$")

        # 8 unigram/bgiram feature
        flist.append("w." + w)

        # prefix/suffix
        for i in range(1, 4):
            if len(w)>=i:
                flist.append("tr1.pre.%d.%s"%(i, w[:i]))
                flist.append("tr1.post.%d.%s"%(i, w[-i:]))
            else:
                flist.append("/")
                flist.append("/")

        if idx > 0:
            flist.append("tr1.w-1." + wordary[idx - 1])
        else:
            flist.append("tr1.w-1.BOS")
        if idx < len(wordary) - 1:
            flist.append("tr1.w1." + wordary[idx + 1])
        else:
            flist.append("tr1.w1.EOS")
        if idx > 1:
            flist.append("tr1.w-2." + wordary[idx - 2])
        else:
            flist.append("tr1.w-2.BOS")
        if idx < len(wordary) - 2:
            flist.append("tr1.w2." + wordary[idx + 2])
        else:
            flist.append("tr1.w2.EOS")
        if idx > 0:
            flist.append("tr1.w_-1_0." + wordary[idx - 1] + "." + w)
        else:
            flist.append("tr1.w_-1_0.BOS")
        if idx < len(wordary) - 1:
            flist.append("tr1.w_0_1." + w + "." + wordary[idx + 1])
        else:
            flist.append("tr1.w_0_1.EOS")

        return flist

    def convert_feature_file_to_idx_file(
        self, feature_file, feature_idx_file, tag_idx_file
    ):

        with open(feature_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        with open(feature_idx_file, "w", encoding="utf8") as f_writer, open(
            tag_idx_file, "w", encoding="utf8"
        ) as t_writer:

            f_writer.write("{}\n\n".format(len(self.feature_to_idx)))
            t_writer.write("{}\n\n".format(len(self.tag_to_idx)))

            tags_idx = []  # type: List[str]
            features_idx = []  # type: List[List[str]]
            for line in lines:
                line = line.strip()
                if not line:
                    # sentence finish
                    for feature_idx in features_idx:
                        if not feature_idx:
                            f_writer.write("0\n")
                        else:
                            f_writer.write(",".join(map(str, feature_idx)))
                            f_writer.write("\n")
                    f_writer.write("\n")

                    t_writer.write(",".join(map(str, tags_idx)))
                    t_writer.write("\n\n")

                    tags_idx = []
                    features_idx = []
                    continue

                splits = line.split(" ")
                feature_idx = [
                    self.feature_to_idx[feat]
                    for feat in splits[:-1]
                    if feat in self.feature_to_idx
                ]
                features_idx.append(feature_idx)
                if not splits[-1] in self.tag_to_idx:
                    tags_idx.append(-1)
                else:
                    tags_idx.append(self.tag_to_idx[splits[-1]])

    def convert_text_file_to_feature_file(
        self, text_file, conll_file=None, feature_file=None
    ):

        if conll_file is None:
            conll_file = "{}.conll{}".format(*os.path.split(text_file))
        if feature_file is None:
            feature_file = "{}.feat{}".format(*os.path.split(text_file))

        conll_line_format = "{} {}\n"

        with open(text_file, "r", encoding="utf8") as reader, open(
            conll_file, "w", encoding="utf8"
        ) as c_writer, open(feature_file, "w", encoding="utf8") as f_writer:
            for line in reader.read().strip().replace("\r", "").split("\n\n"):
                line = line.strip()
                if not line:
                    continue
                line = self.keyword_rename(line).split("\n")
                words = []
                tags = []
                for word_tag in line:
                    word, tag = word_tag.split()
                    words.append(word)
                    tags.append(tag)
                example = [
                    self._num_letter_normalize(word)
                    for word in words
                ]
                for word, tag in zip(example, tags):
                    c_writer.write(conll_line_format.format(word, tag))
                c_writer.write("\n")

                for idx, tag in enumerate(tags):
                    features = self.get_node_features(idx, example)
                    features = [
                        (feature if feature in self.feature_to_idx else "/")
                        for feature in features
                    ]
                    features.append(tag)
                    f_writer.write(" ".join(features))
                    f_writer.write("\n")
                f_writer.write("\n")

    def save(self, model_dir):
        data = {}
        data["feature_to_idx"] = self.feature_to_idx
        data["tag_to_idx"] = self.tag_to_idx

        with open(os.path.join(model_dir, 'features.pkl'), 'wb') as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, model_dir):
        extractor = cls.__new__(cls)

        feature_path = os.path.join(model_dir, "features.pkl")
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as reader:
                data = pickle.load(reader)
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]

            return extractor


        print(
            "WARNING: features.pkl does not exist, try loading features.json",
            file=sys.stderr,
        )


        feature_path = os.path.join(model_dir, "features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf8") as reader:
                data = json.load(reader)
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]
            extractor.save(model_dir)
            return extractor
        print(
            "WARNING: features.json does not exist, try loading using old format",
            file=sys.stderr,
        )

        extractor.feature_to_idx = {}
        feature_base_name = os.path.join(model_dir, "featureIndex.txt")
        for i in range(10):
            with open(
                "{}_{}".format(feature_base_name, i), "r", encoding="utf8"
            ) as reader:
                for line in reader:
                    feature, index = line.split(" ")
                    feature = ".".join(feature.split(".")[1:])
                    extractor.feature_to_idx[feature] = int(index)

        extractor.tag_to_idx = {}
        with open(
            os.path.join(model_dir, "tagIndex.txt"), "r", encoding="utf8"
        ) as reader:
            for line in reader:
                tag, index = line.split(" ")
                extractor.tag_to_idx[tag] = int(index)

        print(
            "INFO: features.json is saved",
            file=sys.stderr,
        )
        extractor.save(model_dir)

        return extractor
