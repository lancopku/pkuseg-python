import numpy as np
import math

class decisionNode:
    def __init__(self):
        self._preY = -1
        self._maxPreScore = -1
        self._maxNowScore = -1
        self._initCheck = False

class Viterbi:
    def __init__(self ,w, h):
        self._w = w
        self._h = h
        self._nodeScore = []
        self._edgeScore = []
        self._decisionLattice = []
        for i in range(w):
            tmp = []
            for j in range(h):
                tmp.append(decisionNode())
            self._decisionLattice.append(tmp)
        for i in range(w):
            self._nodeScore.append(np.zeros(h))
            self._edgeScore.append(np.zeros((h, h)))
        self._edgeScore[0] = None

    def setScores(self, i, Y, YY):
        self._nodeScore[i] = Y.copy()
        if i>0:
            self._edgeScore[i] = YY.copy()

    def runViterbi(self, states, exp):
        for y in range(self._h):
            curNode = self._decisionLattice[-1][y]
            curNode._initCheck = True
            curNode._maxPreScore = 0
            curNode._maxNowScore = self._nodeScore[-1][y]
            curNode._preY = -1
        for i in range(self._w-2, -1, -1):
            for y in range(self._h):
                for yPre in range(self._h):
                    iPre = i+1
                    preNode = self._decisionLattice[iPre][yPre]
                    curNode = self._decisionLattice[i][y]
                    score1 = self._nodeScore[iPre][yPre]
                    score2 = self._edgeScore[iPre][y, yPre]
                    score3 = preNode._maxPreScore
                    score4 = self._nodeScore[i][y]
                    preScore = score1+score2+score3
                    if not curNode._initCheck:
                        curNode._initCheck = True
                        curNode._maxPreScore = preScore
                        curNode._maxNowScore = preScore + score4
                        curNode._preY = yPre
                    elif preScore >= curNode._maxPreScore:
                        curNode._maxPreScore = preScore
                        curNode._maxNowScore = preScore + score4
                        curNode._preY = yPre
        states.clear()
        ma = self._decisionLattice[0][0]._maxNowScore
        tag = 0
        for y in range(1, self._h):
            sc = self._decisionLattice[0][y]._maxNowScore
            if ma<sc:
                ma = sc
                tag = y
        states.append(tag)
        for i in range(1, self._w):
            iPre = i-1
            tag = self._decisionLattice[iPre][tag]._preY
            states.append(tag)
        # overflow
        if ma>300:
            ma = 300
        return math.exp(ma)
