import numpy as np
import gzip
import tqdm

PATH_EMB = "../askubuntu/vector/vectors_pruned.200.txt.gz"
PATH_TEXT = "../askubuntu/text_tokenized.txt.gz"
PATH_DEV = "../askubuntu/dev.txt"
PATH_TEST = "../askubuntu/test.txt"
PATH_TRAIN = "../askubuntu/train_random.txt"

EMB_LEN = 200

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

            max_title = len(title) if self.title_dim < len(title)
            max_body = len(body) if self.body_dim < len(body)

            title2iarr = [self.word2idx[x] if x in self.word2idx for x in title]
            body2iarr = [self.word2idx[x] if x in self.word2idx for x in body]

            if len(title2iarr) != 0 and len(body2iarr) != 0:
                id2data[qid] = (title2iarr, body2iarr)

    return id2data, max_title, max_body


def loadDataset(args):
    print "\nLoading data..."
    embedding_tensor, word2idx = getEmbeddingTensor()
    id2data, max_title, max_body = getId2Data(word2idx)

    args.embedding_dim = embeddings.shape[1]

    train_data = dataset.AskUbuntuDataset(PATH_TRAIN, id2data, max_title, max_body)
    dev_data = dataset.AskUbuntuDataset(PATH_TEST, id2data, max_title, max_body)
    test_data = dataset.AskUbuntuDataset(PATH_DEV, id2data, max_title, max_body)

    return train_data, dev_data, test_data, embedding_tensor
