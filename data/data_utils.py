import numpy as np
import data.dataset as dataset
import gzip
import tqdm
import cPickle as pickle
from zipfile import ZipFile

PATH_EMB = "glove.emb.zip"
EMB_FNAME = "glove.emb"
PATH_TEXT = "./askubuntu/text_tokenized.txt.gz"

PATH_ADEV_NEG = "./Android/dev.neg.txt"
PATH_ADEV_POS = "./Android/dev.pos.txt"
PATH_ATEST_NEG = "./Android/test.neg.txt"
PATH_ATEST_POS = "./Android/test.pos.txt"

PATH_ACORP = "./Android/corpus.tsv.gz"

PATH_EMB_SAVE = './embedding.pickle'
PATH_id2target_SAVE = './id2source.pickle'
PATH_CONST_SAVE = './consts.txt'


EMB_LEN = 300
MAX_BODY_LEN = 100

def getEmbeddingTensor():
    word2idx = {}
    embedding_tensor = []
    embedding_tensor.append(np.zeros(EMB_LEN))
    zipf = ZipFile(PATH_EMB)
    global EMB_LEN

    with zipf.open(EMB_FNAME) as gfile:
        for i, line in enumerate(gfile, start=1):
            word, emb = line.split()[0], line.split()[1:]
            EMB_LEN = len(emb)
            vector = [float(x) for x in emb]
            embedding_tensor.append(vector)
            word2idx[word] = i
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word2idx

def get_id2source(word2idx):
    id2source = {}
    max_title = 0
    max_body = 0

    with gzip.open(PATH_TEXT) as gfile:
        for line in tqdm.tqdm(gfile):
            qid, qtitle, qbody = line.split('\t')
            title = qtitle.split()
            body = qbody.split()

            title2iarr = [word2idx[x] if x in word2idx else 0 for x in title ]
            body2iarr = []

            count = 0
            for word in body:
                if count >= MAX_BODY_LEN:
                    break
                if word in word2idx:
                    body2iarr.append(word2idx[word])
                else:
                    body2iarr.append(0)
                count += 1

            if max_title < len(title2iarr):
                max_title = len(title2iarr)


            if len(title2iarr) != 0 and len(body2iarr) != 0:
                id2source[qid] = ((title2iarr, len(title2iarr)), (body2iarr, len(body2iarr) ))

    return id2source, max_title

def get_id2target(word2idx):
    id2target = {}
    max_title = 0
    max_body = 0

    with gzip.open(PATH_ACORP) as gfile:
        for line in tqdm.tqdm(gfile):
            qid, qtitle, qbody = line.split('\t')
            title = qtitle.split()
            body = qbody.split()

            title2iarr = [word2idx[x] for x in title if x in word2idx]
            body2iarr = []

            count = 0
            for word in body:
                if word in word2idx:
                    if count >= MAX_BODY_LEN:
                        break
                    body2iarr.append(word2idx[word])
                    count += 1

            if max_title < len(title2iarr):
                max_title = len(title2iarr)


            if len(title2iarr) != 0 and len(body2iarr) != 0:
                id2target[qid] = ((title2iarr, len(title2iarr)), (body2iarr, len(body2iarr) ))

    return id2target, max_title

def createTestDic(id2target):
    dic_test = {}

    with open(PATH_ATEST_POS) as f:
        for line in tqdm.tqdm(f):
            first, second= line.split()
            if (first in id2target) and (second in id2target):
                dic_test[first] = []
                dic_test[first].append(second)

    with open(PATH_ATEST_NEG) as f:
        for line in tqdm.tqdm(f):
            first, second= line.split()
            if (first in id2target) and (second in id2target) and (first in dic_test):
                dic_test[first].append(second)

    return dic_test

def createAndroidDics(id2target):
    dic_dev = {}
    domain_question = []

    dic_test = createTestDic(id2target)

    with open(PATH_ADEV_POS) as f:
        for line in tqdm.tqdm(f):
            first, second= line.split()
            if (first in id2target) and (second in id2target):
                dic_dev[first] = []
                dic_dev[first].append(second)

    with open(PATH_ADEV_NEG) as f:
        for line in tqdm.tqdm(f):
            first, second= line.split()
            if (first in id2target) and (second in id2target) and (first in dic_dev):
                dic_dev[first].append(second)

    for q in id2target:
        if q not in dic_test:
            domain_question.append(q)

    return dic_dev, dic_test, domain_question

def loadDataset(args):
    print "\nLoading embedding, train, and dev data..."
    embedding_tensor, word2idx = getEmbeddingTensor()

    id2source, max_title = get_id2source(word2idx)
    id2target, max_title2 = get_id2target(word2idx)
    max_title = max_title2 if max_title2 > max_title else max_title
    args.embedding_dim = embedding_tensor.shape[1]

    dic_dev, dic_test, domain_question = createAndroidDics(id2target)

    train_data = dataset.TrainDataset(id2source, id2target, domain_question, max_title, MAX_BODY_LEN)

    dev_data =  dataset.EvalDataset(id2target, dic_dev, max_title, MAX_BODY_LEN)

    with open(PATH_id2target_SAVE, 'wb') as handle:
        pickle.dump(id2target, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_CONST_SAVE, 'w') as tfile:
        tfile.write(str(max_title) + '\t' + str(MAX_BODY_LEN) + '\t' + str(args.embedding_dim))

    return train_data, dev_data, embedding_tensor


def loadTest(args):
    print "\nLoading paramters and test data..."

    with open(PATH_id2target_SAVE, 'rb') as handle:
        id2target = pickle.load(handle)
    with open(PATH_CONST_SAVE, 'r') as tfile:
        max_title, max_body, args.embedding_dim = [int(x) for x in tfile.read().split()]

    #test data will come from target

    dic_test = createTestDic(id2target)
    test_data =  dataset.EvalDataset(id2target, dic_test, max_title, MAX_BODY_LEN)

    return test_data
