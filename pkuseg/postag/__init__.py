from __future__ import print_function
import sys

import os
import time

from ..inference import decodeViterbi_fast


from .feature_extractor import FeatureExtractor
from .model import Model

class Postag:
    def __init__(self, model_name):
        modelDir = model_name
        self.feature_extractor = FeatureExtractor.load(modelDir)
        self.model = Model.load(modelDir)

        self.idx_to_tag = {
            idx: tag for tag, idx in self.feature_extractor.tag_to_idx.items()
        }

        self.n_feature = len(self.feature_extractor.feature_to_idx)
        self.n_tag = len(self.feature_extractor.tag_to_idx)

        # print("finish")

    def _cut(self, text):
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

        _, tags = decodeViterbi_fast(all_feature, self.model)
        tags = list(map(lambda x:self.idx_to_tag[x], tags))
        return tags

    def tag(self, sen):
        """txt: list[str], tags: list[str]"""
        tags = self._cut(sen)
        return tags

