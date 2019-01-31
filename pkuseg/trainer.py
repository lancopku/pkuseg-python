# from .config import config
# from .feature import *
# from .data_format import *
# from .toolbox import *
import os
import time
from multiprocessing import Process, Queue

from pkuseg import res_summarize

# from .inference import *
# from .config import Config
from pkuseg.config import Config, config
from pkuseg.data import DataSet
from pkuseg.feature_extractor import FeatureExtractor

# from .feature_generator import *
from pkuseg.model import Model
import pkuseg.inference as _inf

# from .inference import *
# from .gradient import *
from pkuseg.optimizer import ADF
from pkuseg.scorer import getFscore

# from typing import TextIO

# from .res_summarize import summarize
# from .res_summarize import write as reswrite

# from pkuseg.trainer import Trainer


def train(config=None):
    if config is None:
        config = Config()

    if config.init_model is None:
        feature_extractor = FeatureExtractor()
    else:
        feature_extractor = FeatureExtractor.load(config.init_model)
    feature_extractor.build(config.trainFile)
    feature_extractor.save()

    feature_extractor.convert_text_file_to_feature_file(
        config.trainFile, config.c_train, config.f_train
    )
    feature_extractor.convert_text_file_to_feature_file(
        config.testFile, config.c_test, config.f_test
    )

    feature_extractor.convert_feature_file_to_idx_file(
        config.f_train, config.fFeatureTrain, config.fGoldTrain
    )
    feature_extractor.convert_feature_file_to_idx_file(
        config.f_test, config.fFeatureTest, config.fGoldTest
    )

    config.globalCheck()

    config.swLog = open(os.path.join(config.outDir, config.fLog), "w")
    config.swResRaw = open(os.path.join(config.outDir, config.fResRaw), "w")
    config.swTune = open(os.path.join(config.outDir, config.fTune), "w")

    print("\nstart training...")
    config.swLog.write("\nstart training...\n")

    print("\nreading training & test data...")
    config.swLog.write("\nreading training & test data...\n")

    trainset = DataSet.load(config.fFeatureTrain, config.fGoldTrain)
    testset = DataSet.load(config.fFeatureTest, config.fGoldTest)

    trainset = trainset.resize(config.trainSizeScale)

    print(
        "done! train/test data sizes: {}/{}".format(len(trainset), len(testset))
    )
    config.swLog.write(
        "done! train/test data sizes: {}/{}\n".format(
            len(trainset), len(testset)
        )
    )

    config.swLog.write("\nr: {}\n".format(config.reg))
    print("\nr: {}".format(config.reg))
    if config.rawResWrite:
        config.swResRaw.write("\n%r: {}\n".format(config.reg))

    trainer = Trainer(config, trainset, feature_extractor)

    time_list = []
    err_list = []
    diff_list = []
    score_list_list = []

    for i in range(config.ttlIter):
        # config.glbIter += 1
        time_s = time.time()
        err, sample_size, diff = trainer.train_epoch()
        time_t = time.time() - time_s
        time_list.append(time_t)
        err_list.append(err)
        diff_list.append(diff)

        score_list = trainer.test(testset, i)
        score_list_list.append(score_list)
        score = score_list[0]

        logstr = "iter{}  diff={:.2e}  train-time(sec)={:.2f}  {}={:.2f}%".format(
            i, diff, time_t, config.metric, score
        )
        config.swLog.write(logstr + "\n")
        config.swLog.write("------------------------------------------------\n")
        config.swLog.flush()
        print(logstr)

    res_summarize.write(config, time_list, err_list, diff_list, score_list_list)
    if config.save == 1:
        trainer.model.save()

    config.swLog.close()
    config.swResRaw.close()
    config.swTune.close()

    res_summarize.summarize(config)

    print("finished.")


class Trainer:
    def __init__(self, config, dataset, feature_extractor):
        self.config = config
        self.X = dataset
        self.n_feature = dataset.n_feature
        self.n_tag = dataset.n_tag

        if config.init_model is None:
            self.model = Model(self.n_feature, self.n_tag)
        else:
            self.model = Model.load(config.init_model)
            self.model.expand(self.n_feature, self.n_tag)

        self.optim = self._get_optimizer(dataset, self.model)

        self.feature_extractor = feature_extractor
        self.idx_to_chunk_tag = {}
        for tag, idx in feature_extractor.tag_to_idx.items():
            if tag.startswith("I"):
                tag = "I"
            if tag.startswith("O"):
                tag = "O"
            self.idx_to_chunk_tag[idx] = tag

    def _get_optimizer(self, dataset, model):
        config = self.config
        if "adf" in config.modelOptimizer:
            return ADF(config, dataset, model)

        raise ValueError("Invalid Optimizer")

    def train_epoch(self):
        return self.optim.optimize()

    def test(self, testset, iteration):

        outfile = os.path.join(config.outDir, config.fOutput.format(iteration))

        func_mapping = {
            "tok.acc": self._decode_tokAcc,
            "str.acc": self._decode_strAcc,
            "f1": self._decode_fscore,
        }

        with open(outfile, "w", encoding="utf8") as writer:
            score_list = func_mapping[config.evalMetric](
                testset, self.model, writer
            )

        for example in testset:
            example.predicted_tags = None

        return score_list

    def _decode(self, testset: DataSet, model: Model):
        if config.nThread == 1:
            self._decode_single(testset, model)
        else:
            self._decode_multi_proc(testset, model)

    def _decode_single(self, testset: DataSet, model: Model):
        # n_tag = model.n_tag
        for example in testset:
            _, tags = _inf.decodeViterbi_fast(example.features, model)
            example.predicted_tags = tags

    @staticmethod
    def _decode_proc(model, in_queue, out_queue):
        while True:
            item = in_queue.get()
            if item is None:
                return
            idx, features = item
            _, tags = _inf.decodeViterbi_fast(features, model)
            out_queue.put((idx, tags))

    def _decode_multi_proc(self, testset: DataSet, model: Model):
        in_queue = Queue()
        out_queue = Queue()
        procs = []
        nthread = self.config.nThread
        for i in range(nthread):
            p = Process(
                target=self._decode_proc, args=(model, in_queue, out_queue)
            )
            procs.append(p)

        for idx, example in enumerate(testset):
            in_queue.put((idx, example.features))

        for proc in procs:
            in_queue.put(None)
            proc.start()

        for _ in range(len(testset)):
            idx, tags = out_queue.get()
            testset[idx].predicted_tags = tags

        for p in procs:
            p.join()

    # token accuracy
    def _decode_tokAcc(self, dataset, model, writer):
        config = self.config

        self._decode(dataset, model)
        n_tag = model.n_tag
        all_correct = [0] * n_tag
        all_pred = [0] * n_tag
        all_gold = [0] * n_tag

        for example in dataset:
            pred = example.predicted_tags
            gold = example.tags

            if writer is not None:
                writer.write(",".join(map(str, pred)))
                writer.write("\n")

            for pred_tag, gold_tag in zip(pred, gold):
                all_pred[pred_tag] += 1
                all_gold[gold_tag] += 1
                if pred_tag == gold_tag:
                    all_correct[gold_tag] += 1

        config.swLog.write(
            "% tag-type  #gold  #output  #correct-output  token-precision  token-recall  token-f-score\n"
        )
        sumGold = 0
        sumOutput = 0
        sumCorrOutput = 0

        for i, (correct, gold, pred) in enumerate(
            zip(all_correct, all_gold, all_pred)
        ):
            sumGold += gold
            sumOutput += pred
            sumCorrOutput += correct

            if gold == 0:
                rec = 0
            else:
                rec = correct * 100.0 / gold

            if pred == 0:
                prec = 0
            else:
                prec = correct * 100.0 / pred

            config.swLog.write(
                "% {}:  {}  {}  {}  {:.2f}  {:.2f}  {:.2f}\n".format(
                    i,
                    gold,
                    pred,
                    correct,
                    prec,
                    rec,
                    (2 * prec * rec / (prec + rec)),
                )
            )

        if sumGold == 0:
            rec = 0
        else:
            rec = sumCorrOutput * 100.0 / sumGold
        if sumOutput == 0:
            prec = 0
        else:
            prec = sumCorrOutput * 100.0 / sumOutput

        if prec == 0 and rec == 0:
            fscore = 0
        else:
            fscore = 2 * prec * rec / (prec + rec)

        config.swLog.write(
            "% overall-tags:  {}  {}  {}  {:.2f}  {:.2f}  {:.2f}\n".format(
                sumGold, sumOutput, sumCorrOutput, prec, rec, fscore
            )
        )
        config.swLog.flush()
        return [fscore]

    def _decode_strAcc(self, dataset, model, writer):

        config = self.config

        self._decode(dataset, model)

        correct = 0
        total = len(dataset)

        for example in dataset:
            pred = example.predicted_tags
            gold = example.tags

            if writer is not None:
                writer.write(",".join(map(str, pred)))
                writer.write("\n")

            for pred_tag, gold_tag in zip(pred, gold):
                if pred_tag != gold_tag:
                    break
            else:
                correct += 1

        acc = correct / total * 100.0
        config.swLog.write(
            "total-tag-strings={}  correct-tag-strings={}  string-accuracy={}%".format(
                total, correct, acc
            )
        )
        return [acc]

    def _decode_fscore(self, dataset, model, writer):
        config = self.config

        self._decode(dataset, model)

        gold_tags = []
        pred_tags = []

        for example in dataset:
            pred = example.predicted_tags
            gold = example.tags

            pred_str = ",".join(map(str, pred))
            pred_tags.append(pred_str)
            if writer is not None:
                writer.write(pred_str)
                writer.write("\n")
            gold_tags.append(",".join(map(str, gold)))

        scoreList, infoList = getFscore(
            gold_tags, pred_tags, self.idx_to_chunk_tag
        )
        config.swLog.write(
            "#gold-chunk={}  #output-chunk={}  #correct-output-chunk={}  precision={:.2f}  recall={:.2f}  f-score={:.2f}\n".format(
                infoList[0],
                infoList[1],
                infoList[2],
                scoreList[1],
                scoreList[2],
                scoreList[0],
            )
        )
        return scoreList

    #     acc = correct / total * 100.0
    #     config.swLog.write(
    #         "total-tag-strings={}  correct-tag-strings={}  string-accuracy={}%".format(
    #             total, correct, acc
    #         )
    #     )

    #     goldTagList = []
    #     resTagList = []
    #     for x in X2:
    #         res = ""
    #         for im in x._yOutput:
    #             res += str(im) + ","
    #         resTagList.append(res)
    #         # if not dynamic:
    #         if writer is not None:
    #             for i in range(len(x._yOutput)):
    #                 writer.write(str(x._yOutput[i]) + ",")
    #             writer.write("\n")
    #         goldTags = x._x.getTags()
    #         gold = ""
    #         for im in goldTags:
    #             gold += str(im) + ","
    #         goldTagList.append(gold)
    #     # if dynamic:
    #     #     return resTagList
    #     scoreList = []

    #     if config.runMode == "train":
    #         infoList = []
    #         scoreList = getFscore(
    #             goldTagList, resTagList, infoList, self.idx_to_chunk_tag
    #         )
    #         config.swLog.write(
    #             "#gold-chunk={}  #output-chunk={}  #correct-output-chunk={}  precision={:.2f}  recall={:.2f}  f-score={:.2f}\n".format(
    #                 infoList[0],
    #                 infoList[1],
    #                 infoList[2],
    #                 "%.2f" % scoreList[1],
    #                 "%.2f" % scoreList[2],
    #                 "%.2f" % scoreList[0],
    #             )
    #         )
    #     return scoreList

    # # def multiThreading(self, X, X2):
    #     config = self.config
    #     # if dynamic:
    #     #     for i in range(len(X)):
    #     #         X2.append(dataSeqTest(X[i], []))
    #     #     for k, x in enumerate(X2):
    #     #         tags = []
    #     #         prob = self.Inf.decodeViterbi_fast(self.Model, x._x, tags)
    #     #         X2[k]._yOutput.clear()
    #     #         X2[k]._yOutput.extend(tags)
    #     #     return

    #     for i in range(len(X)):
    #         X2.append(dataSeqTest(X[i], []))
    #     if len(X) < config.nThread:
    #         config.nThread = len(X)
    #     interval = (len(X2) + config.nThread - 1) // config.nThread
    #     procs = []
    #     Q = Queue(5000)
    #     for i in range(config.nThread):
    #         start = i * interval
    #         end = min(start + interval, len(X2))
    #         proc = Process(
    #             target=Trainer.taskRunner_test,
    #             args=(self.Inf, self.Model, X2, start, end, Q),
    #         )
    #         proc.start()
    #         procs.append(proc)
    #     for i in range(len(X2)):
    #         t = Q.get()
    #         k, tags = t
    #         X2[k]._yOutput.clear()
    #         X2[k]._yOutput.extend(tags)
    #     for proc in procs:
    #         proc.join()

    # @staticmethod
    # def taskRunner_test(Inf, Model, X2, start, end, Q):
    #     for k in range(start, end):
    #         x = X2[k]
    #         tags = []
    #         prob = Inf.decodeViterbi_fast(Model, x._x, tags)
    #         Q.put((k, tags))
