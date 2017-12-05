import data_utils as du
import torch
import gzip
import tqdm
import torch.utils.data as data
import random

PATH_TRAIN = "./askubuntu/train_random.txt"

MAX_QCOUNT = 20
size = 0

def pad(arr, l):
    while len(arr) < l:
            arr.append(0)

def padmask(origlen, maxlen):
    m = []

    for i in range(maxlen):
        if i < origlen:
            m.append(1)
        else:
            m.append(0)
    return m

def evalSampleDic(q_title, p_title, q_body, p_body, qt_mask, pt_mask, qb_mask, pb_mask):
    sample = {
    'titles': [q_title, p_title],
    'bodies':[q_body, p_body],
    'titles_masks':[qt_mask, pt_mask],
    'bodies_masks':[qb_mask, pb_mask]
    }
    return sample


def trainSampleDic(id2source_list, id2source, id2target,
                    domain_question, q_title, p_title, q_body,
                    p_body, qt_mask, pt_mask, qb_mask, pb_mask, max_title, max_body):

    samples = {
    'titles': [q_title, p_title],
    'bodies':[q_body, p_body],
    'titles_masks':[qt_mask, pt_mask],
    'bodies_masks':[qb_mask, pb_mask]
    }

    label = random.choice([0,1])
    if label == 0:
        choice = random.choice(id2source_list)
        (title, t_mask), (body, b_mask) = id2source[choice]
    else:
        choice = random.choice(domain_question)
        while choice not in domain_question:
            choice = random.choice(domain_question)
        (title, t_mask), (body, b_mask) = id2target[choice]

    t_mask = padmask(t_mask, max_title)
    b_mask = padmask(b_mask, max_body)

    pad(title, max_title)
    pad(body, max_body)

    question = {
    'bodies': [body],
    'titles': [title],
    'bodies_masks': [b_mask],
    'titles_masks': [t_mask],
    'domain': [label]
    }

    sample = {
    'source_samples': samples,
    'question': question
    }

    return sample


def processCandidate(sample, candidate, id2target, max_title, max_body, isTrain):
    (title, t_mask), (body, b_mask) = id2target[candidate]

    t_mask = padmask(t_mask, max_title)
    b_mask = padmask(b_mask, max_body)

    pad(title, max_title)
    pad(body, max_body)


    if isTrain:
        sampl = sample['source_samples']
    else:
        sampl = sample

    sampl['titles'].append(title)
    sampl['bodies'].append(body)
    sampl['titles_masks'].append(t_mask)
    sampl['bodies_masks'].append(b_mask)


def getCandidate(uset, id2data_list):
    while True:
        candidate = random.choice(id2data_list)
        if candidate not in uset:
            break
    return candidate

def fillInSample(sample, p, q, negs, pos, id2data, id2data_list, max_title, max_body, isTrain):
    count_negs = 0

    while count_negs < MAX_QCOUNT-1:
        neg_candidate = None
        if count_negs < len(negs):
            neg_candidate = negs[count_negs]

        value = id2data.get(neg_candidate)

        if count_negs >= len(negs) or value == None:
            neg_candidate = getCandidate(set().union(negs, pos, [q]), id2data_list)

        if isTrain:
            processCandidate(sample, neg_candidate, id2data, max_title, max_body, True)
        else:
            processCandidate(sample, neg_candidate, id2data, max_title, max_body, False)

        count_negs += 1

def createSample(q, p, negs, pos, id2source, id2target, id2source_list, id2target_list, domain_question,  max_title, max_body, isTrain):
    id2data = id2source

    if isTrain:
        if q not in id2source or p not in id2source:
            return None
    else:
        id2data = id2target
        if q not in id2target or p not in id2target:
            return None

    (q_title, qt_mask), (q_body, qb_mask) = id2data[q]
    qt_mask = padmask(qt_mask, max_title)
    qb_mask = padmask(qb_mask, max_body)
    pad(q_title, max_title)
    pad(q_body, max_body)

    (p_title, pt_mask), (p_body, pb_mask) = id2data[p]
    pt_mask = padmask(pt_mask, max_title)
    pb_mask = padmask(pb_mask, max_body)
    pad(p_title, max_title)
    pad(p_body, max_body)

    if isTrain:
        sample = trainSampleDic(id2source_list, id2source, id2target,
                            domain_question, q_title, p_title, q_body,
                            p_body, qt_mask, pt_mask, qb_mask, pb_mask, max_title, max_body)
    else:
        #sample = evalSampleDic(q_title, q_body, qt_mask, qb_mask)
        sample = evalSampleDic(q_title, p_title, q_body, p_body, qt_mask, pt_mask, qb_mask, pb_mask)

    random.shuffle(negs)

    if isTrain:
        fillInSample(sample, p, q, negs, pos, id2source, id2source_list, max_title, max_body, isTrain)
    else:
        fillInSample(sample, p, q, negs, pos, id2target, id2target_list, max_title, max_body, isTrain)


    if isTrain:
        samples = sample['source_samples']
        question = sample['question']

        for key in samples:
            samples[key] = torch.LongTensor(samples[key])
        for key in question:
            question[key] = torch.LongTensor(question[key])

    else:
        for key in sample:
            sample[key] = torch.LongTensor(sample[key])

    return sample


class EvalDataset(data.Dataset):
    def __init__(self, id2target, dic, max_title, max_body):
        self.dataset = []
        self.id2target = id2target
        self.id2target_list = list(id2target)
        self.dic = dic

        for key in dic:
            if len(dic[key]) >= 20:
                sample = createSample(key, dic[key][0], dic[key][1:], None, None,
                                      id2target, None, self.id2target_list, None, max_title, max_body, False)
                if sample != None:
                    self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

class TrainDataset(data.Dataset):
    def __init__(self, id2source, dic_dev, id2target, domain_question, max_title, max_body):
        self.dataset = []
        self.id2source = id2source
        self.id2target = id2target
        self.id2target_list = list(id2target)  #for random pick
        self.id2source_list = list(id2source)  #for random pick
        self.dic = dic_dev

        keys = list(dic_dev.keys())

        with open(PATH_TRAIN) as f:
            for line in f:
                split = line.split('\t')
                q = split[0]
                pos = split[1].split()
                negs = split[2].split()

                for p in pos:
                    sample = createSample(q, p, negs, pos, id2source, id2target,
                                        self.id2source_list, self.id2target_list, domain_question,
                                        max_title, max_body, True)
                    if sample != None:
                        target_sample = None
                        while target_sample == None:
                            key = random.choice(keys)
                            target_sample = createSample(key, dic_dev[key][0], dic_dev[key][1:], None, None,
                                              id2target, None, self.id2target_list, None, max_title, max_body, False)

                        sample.update({"target_samples": target_sample})

                        self.dataset.append(sample)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample
