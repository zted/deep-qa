infile = '/home/ted/COE/reranker/data/submission.txt'
outfile = '/home/ted/COE/reranker/data/submission_sorted.txt'


def writeOut(somefile, someArray):
    sortedArray = sorted(someArray, key=lambda score: score[0], reverse=True)
    for a in sortedArray:
        somefile.write('\t'.join(a[1]))
    return


outf = open(outfile, 'w')
with open(infile, 'r') as f:
    previous_id = '0'
    first = True
    someArray = []
    for line in f:
        splits = line.split(' ')
        ID = splits[0]
        score = float(splits[4])
        if ID != previous_id and not first:
            writeOut(outf, someArray)
            someArray = []
        someTup = (score, splits)
        someArray.append(someTup)
        first = False
        previous_id = ID
    writeOut(outf,someArray)
outf.close()