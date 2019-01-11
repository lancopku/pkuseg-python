import os

def tocrfoutput(config, readpath, writedatapath, rawdatapath):
    with open(os.path.join(config.modelDir, "tagIndex.txt")) as tagfile:
        lines = tagfile.readlines()
    tags = {}
    for line in lines:
        wordTags = line.split(" ")
        tags[int(wordTags[1])] = wordTags[0]

    with open(readpath, encoding="utf-8") as outputtag, open(
        writedatapath, "w", encoding="utf-8"
    ) as sw, open(rawdatapath, encoding="utf-8") as rawtext:
        lines = outputtag.readlines()
        rawlines = rawtext.readlines()
        for line, raw in zip(lines, rawlines):
            write_string = ""
            linetag = line.split(",")
            for i, word in enumerate(raw):
                if linetag[i] == "\n":
                    continue
                if tags[int(linetag[i])].find("B") >= 0:
                    write_string = write_string + " " + word
                else:
                    write_string = write_string + word
            sw.write(write_string.strip() + "\n")
