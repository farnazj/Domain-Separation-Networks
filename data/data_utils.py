import numpy as np
import data.dataset as dataset
import gzip
import tqdm
import cPickle as pickle

PATH_EMB = "./askubuntu/vector/vectors_pruned.200.txt.gz"
PATH_TEXT = "./askubuntu/text_tokenized.txt.gz"
PATH_DEV = "./askubuntu/dev.txt"
PATH_TEST = "./askubuntu/test.txt"
PATH_TRAIN = "./askubuntu/train_random.txt"

PATH_EMB_SAVE = './embedding.pickle'
PATH_ID2DATA_SAVE = './id2data.pickle'
PATH_CONST_SAVE = './consts.txt'


EMB_LEN = 200
MAX_BODY_LEN = 100

def getEmbeddingTensor():
    word2idx = {}
    embedding_tensor = []
    embedding_tensor.append(np.zeros(EMB_LEN))

    with gzip.open(PATH_EMB) as gfile:
        for i, line in enumerate(gfile, start=1):
            word, emb = line.split()[0], line.split()[1:]
            vector = [float(x) for x in emb]
            embedding_tensor.append(vector)
            word2idx[word] = i
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word2idx

def getId2Data(word2idx):
    id2data = {}
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
                id2data[qid] = ((title2iarr, len(title2iarr)), (body2iarr, len(body2iarr) ))

    return id2data, max_title, MAX_BODY_LEN


def loadDataset(args):
    print "\nLoading embedding, train, and dev data..."
    embedding_tensor, word2idx = getEmbeddingTensor()
    id2data, max_title, max_body = getId2Data(word2idx)

    args.embedding_dim = embedding_tensor.shape[1]

    train_data = dataset.AskUbuntuDataset(PATH_TRAIN, id2data, max_title, max_body, True)
    dev_data = dataset.AskUbuntuDataset(PATH_DEV, id2data, max_title, max_body, False)

    with open(PATH_ID2DATA_SAVE, 'wb') as handle:
        pickle.dump(id2data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_CONST_SAVE, 'w') as tfile:
        tfile.write(str(max_title) + '\t' + str(max_body) + '\t' + str(args.embedding_dim))

    return train_data, dev_data, embedding_tensor


def loadTest(args):
    print "\nLoading paramters and test data..."

    with open(PATH_ID2DATA_SAVE, 'rb') as handle:
        id2data = pickle.load(handle)
    with open(PATH_CONST_SAVE, 'r') as tfile:
        max_title, max_body, args.embedding_dim = [int(x) for x in tfile.read().split()]

    test_data = dataset.AskUbuntuDataset(PATH_TEST, id2data, max_title, max_body, False)

    return test_data
