import random

import numpy as np
import pkuseg.gradient as _grad

# from pkuseg.config import config


class Optimizer:
    def __init__(self):
        self._preVals = []

    def converge_test(self, err):
        val = 1e100
        if len(self._preVals) > 1:
            prevVal = self._preVals[0]
            if len(self._preVals) == 10:
                self._preVals.pop(0)
            avgImprovement = (prevVal - err) / len(self._preVals)
            relAvg = avgImprovement / abs(err)
            val = relAvg
        self._preVals.append(err)
        return val

    def optimize(self):
        raise NotImplementedError()


class ADF(Optimizer):
    def __init__(self, config, dataset, model):

        super().__init__()

        self.config = config

        self._model = model
        self._X = dataset
        self.decayList = np.ones_like(self._model.w) * config.rate0

    def optimize(self):
        config = self.config
        sample_size = 0
        w = self._model.w
        fsize = w.shape[0]
        xsize = len(self._X)
        grad = np.zeros(fsize)
        error = 0

        feature_count_list = np.zeros(fsize)
        # feature_count_list = [0] * fsize
        ri = list(range(xsize))
        random.shuffle(ri)

        update_interval = xsize // config.nUpdate

        # config.interval = xsize // config.nUpdate
        n_sample = 0
        for t in range(0, xsize, config.miniBatch):
            XX = []
            end = False
            for k in range(t, t + config.miniBatch):
                i = ri[k]
                x = self._X[i]
                XX.append(x)
                if k == xsize - 1:
                    end = True
                    break
            mb_size = len(XX)
            n_sample += mb_size

            # fSet = set()

            err, feature_set = _grad.get_grad_SGD_minibatch(
                grad, self._model, XX
            )
            error += err

            feature_set = list(feature_set)

            feature_count_list[feature_set] += 1

            # for i in feature_set:
            #     feature_count_list[i] += 1
            check = False

            for k in range(t, t + config.miniBatch):
                if t != 0 and k % update_interval == 0:
                    check = True

            # update decay rates
            if check or end:

                self.decayList *= (
                    config.upper
                    - (config.upper - config.lower)
                    * feature_count_list
                    / n_sample
                )
                feature_count_list.fill(0)

                # for i in range(fsize):
                #     v = feature_count_list[i]
                #     u = v / n_sample
                #     eta = config.upper - (config.upper - config.lower) * u
                #     self.decayList[i] *= eta
                # feature_count_list
                # for i in range(len(feature_count_list)):
                #     feature_count_list[i] = 0
            # update weights

            w[feature_set] -= self.decayList[feature_set] * grad[feature_set]
            grad[feature_set] = 0
            # for i in feature_set:
            #     w[i] -= self.decayList[i] * grad[i]
            #     grad[i] = 0
            # reg
            if check or end:
                if config.reg != 0:
                    w -= self.decayList * (
                        w / (config.reg * config.reg) * n_sample / xsize
                    )

                    # for i in range(fsize):
                    #     grad_i = (
                    #         w[i] / (config.reg * config.reg) * (n_sample / xsize)
                    #     )
                    #     w[i] -= self.decayList[i] * grad_i
                n_sample = 0
            sample_size += mb_size
        if config.reg != 0:
            s = (w * w).sum()
            error += s / (2.0 * config.reg * config.reg)
        diff = self.converge_test(error)
        return error, sample_size, diff
