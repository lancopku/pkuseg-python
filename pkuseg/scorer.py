from pkuseg.config import Config


def getFscore(goldTagList, resTagList, idx_to_chunk_tag):
    scoreList = []
    assert len(resTagList) == len(goldTagList)
    getNewTagList(idx_to_chunk_tag, goldTagList)
    getNewTagList(idx_to_chunk_tag, resTagList)
    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0
    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(Config.comma)
        tmp = []
        for t in resChunkAry:
            if len(t) > 0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(Config.comma)
        tmp = []
        for t in goldChunkAry:
            if len(t) > 0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1
    pre = correct_chunk / res_chunk * 100
    rec = correct_chunk / gold_chunk * 100
    f1 = 0 if correct_chunk == 0 else 2 * pre * rec / (pre + rec)
    scoreList.append(f1)
    scoreList.append(pre)
    scoreList.append(rec)
    infoList = []
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    return scoreList, infoList


def getNewTagList(tagMap, tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        for i in range(len(tagAry)):
            if tagAry[i] == "":
                continue
            index = int(tagAry[i])
            if not index in tagMap:
                raise Exception("Error")
            tagAry[i] = tagMap[index]
        newTags = ",".join(tagAry)
        tmpList.append(newTags)
    tagList.clear()
    for im in tmpList:
        tagList.append(im)


def getChunks(tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        tmp = []
        for t in tagAry:
            if t != "":
                tmp.append(t)
        tagAry = tmp
        chunks = ""
        for i in range(len(tagAry)):
            if tagAry[i].startswith("B"):
                pos = i
                length = 1
                ty = tagAry[i]
                for j in range(i + 1, len(tagAry)):
                    if tagAry[j] == "I":
                        length += 1
                    else:
                        break
                chunk = ty + "*" + str(length) + "*" + str(pos)
                chunks = chunks + chunk + ","
        tmpList.append(chunks)
    return tmpList
