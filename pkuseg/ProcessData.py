
def tocrfoutput(config, readpath, writedatapath, rawdatapath):
    with open(config.modelDir + '/tagIndex.txt') as f:
        lines = f.readlines()
    tags = {}
    for line in lines:
        wordTags = line.split(' ')
        tags[int(wordTags[1])] = wordTags[0]

    outputtag = open(readpath, encoding='utf-8')
    sw = open(writedatapath, 'w', encoding='utf-8')
    rawtext = open(rawdatapath, encoding='utf-8')

    lines = outputtag.readlines()
    rawlines = rawtext.readlines()
    for line, raw in zip(lines, rawlines):
        write_string = ''
        linetag = line.split(',')
        for i in range(len(raw)):
            if linetag[i]=='\n':
                continue
            if tags[int(linetag[i])].find('B')>=0:
                write_string = write_string + ' ' + raw[i]
            else:
                write_string = write_string + raw[i]
        sw.write(write_string.strip()+'\n')
    sw.close()
    outputtag.close()
    rawtext.close()
    
