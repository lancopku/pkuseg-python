import os
import sys
import numpy as np

class Model:
    def __init__(self, n_feature, n_tag):

        self.n_tag = n_tag
        self.n_feature = n_feature
        self.n_transition_feature = n_tag * (n_feature + n_tag)
        self.w = np.zeros(self.n_transition_feature)

    def _get_node_tag_feature_id(self, feature_id, tag_id):
        return feature_id * self.n_tag + tag_id

    def _get_tag_tag_feature_id(self, pre_tag_id, tag_id):
        return self.n_feature * self.n_tag + tag_id * self.n_tag + pre_tag_id

    @classmethod
    def load(cls, model_dir):
        model_path = os.path.join(model_dir, "weights.npz")
        if os.path.exists(model_path):
            npz = np.load(model_path)
            sizes = npz["sizes"]
            w = npz["w"]
            model = cls.__new__(cls)
            model.n_tag = int(sizes[0])
            model.n_feature = int(sizes[1])
            model.n_transition_feature = model.n_tag * (
                model.n_feature + model.n_tag
            )
            model.w = w
            assert model.w.shape[0] == model.n_transition_feature
            return model

        print(
            "WARNING: weights.npz does not exist, try loading using old format",
            file=sys.stderr,
        )

        model_path = os.path.join(model_dir, "model.txt")
        with open(model_path, encoding="utf-8") as f:
            ary = f.readlines()

        model = cls.__new__(cls)
        model.n_tag = int(ary[0].strip())
        wsize = int(ary[1].strip())
        w = np.zeros(wsize)
        for i in range(2, wsize):
            w[i - 2] = float(ary[i].strip())
        model.w = w
        model.n_feature = wsize // model.n_tag - model.n_tag
        model.n_transition_feature = wsize

        model.save(model_dir)
        return model

    @classmethod
    def new(cls, model, copy_weight=True):

        new_model = cls.__new__(cls)
        new_model.n_tag = model.n_tag
        if copy_weight:
            new_model.w = model.w.copy()
        else:
            new_model.w = np.zeros_like(model.w)
        new_model.n_feature = (
            new_model.w.shape[0] // new_model.n_tag - new_model.n_tag
        )
        new_model.n_transition_feature = new_model.w.shape[0]
        return new_model

    def save(self, model_dir):
        sizes = np.array([self.n_tag, self.n_feature])
        np.savez(
            os.path.join(model_dir, "weights.npz"), sizes=sizes, w=self.w
        )
        # np.save
        # with open(file, "w", encoding="utf-8") as f:
        #     f.write("{}\n{}\n".format(self.n_tag, self.w.shape[0]))
        #     for value in self.w:
        #         f.write("{:.4f}\n".format(value))
