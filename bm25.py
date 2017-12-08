import itertools
from operator import itemgetter

PATH_DEV = "./askubuntu/dev.txt"
PATH_TEST = "./askubuntu/test.txt"

def getSimilarIdx(allq, pos):
    arr = []
    for i, q in enumerate(allq):
        if q in pos:
            arr.append(i)
    return arr

def updateScores(similar_indices, bm25, sum_av_prec, sum_ranks, num_samples, top_5, top_1):
    count = 0.0
    last_index = -1
    sum_prec = 0.0
    flag = 0

    scores_list = []

    for j in range(len(bm25)):
        scores_list.append( (bm25[j], j) )

    scores_list = sorted(scores_list, reverse = True, key=itemgetter(0))


    for j in range(len(bm25)):
        if scores_list[j][1] in similar_indices:
            count += 1
            sum_prec += count/(j+1)
            last_index = j+1

            if flag == 0:
                sum_ranks += 1.0/(j+1)
                flag = 1

            if j == 0:
                top_1 += 1

            if j < 5:
                top_5 += 1

    if last_index > 0:
        sum_prec /= count

    sum_av_prec += sum_prec
    num_samples += 1

    return sum_av_prec, sum_ranks, num_samples, top_5, top_1

def computeBM25(path):
    with open(path) as f:
        sum_av_prec = 0.0
        sum_ranks = 0.0
        num_samples = 0.0
        top_5 = 0.0
        top_1 = 0.0

        for line in f:
            q, pos, allq, bm25 = line.split('\t')
            pos = pos.split()
            allq = allq.split()
            bm25 = bm25.split()
            bm25 = [float(x) for x in bm25]
            similar_idx = getSimilarIdx(allq, pos)
            sum_av_prec, sum_ranks, num_samples, top_5, top_1 = \
            updateScores(similar_idx, bm25, sum_av_prec, sum_ranks, num_samples, top_5, top_1)


    print "DEV BM25 MEASURES"
    _map = sum_av_prec/num_samples
    _mrr = sum_ranks/num_samples
    _pat5 = top_5/(num_samples*5)
    _pat1 = top_1/num_samples
    print('MAP: {:.3f}'.format(_map))
    print('MRR: {:.3f}'.format(_mrr))
    print('P@1: {:.3f}'.format(_pat1))
    print('P@5: {:.3f}'.format(_pat5))
