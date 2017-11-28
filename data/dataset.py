import data_utils as du
import torch
import gzip
import tqdm
import torch.utils.data as data
import random

MAX_QCOUNT = 20
MAX_POS = 10
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

class AskUbuntuDataset(data.Dataset):
    def __init__(self, path, id2data, max_title, max_body, isTrain):
        self.path = path
        self.dataset = []
        self.id2data = id2data
        self.title_dim = max_title
        self.body_dim = max_body
        self.id2data_list = list(id2data)

        with open(path) as f:
            for line in tqdm.tqdm(f):
                split = line.split('\t')
                q = split[0]
                pos = split[1].split()
                negs = split[2].split()

                if isTrain:
                    negs = [x for x in negs if x not in pos]

                    for p in pos:
                        sample = self.createTrainSample(q, p, negs, pos, max_title, max_body)
                        if sample != None:
                            self.dataset.append(sample)
                else:
                    if len(pos) > 0:
                        sample = self.createEvalSample(q, negs, pos, max_title, max_body)
                    if sample != None:
                        self.dataset.append(sample)

    def createEvalSample(self, query_q, rest_qs, pos, max_title, max_body):

        if query_q not in self.id2data:
            return None

        (q_title, qt_mask), (q_body, qb_mask) = self.id2data[query_q]

        qt_mask = pad_mask(qt_mask, max_title)
        qb_mask = pad_mask(qb_mask, max_body)
        
        pad(q_title, title_len)
        pad(q_body, body_len)

        sample = {
        'titles': [q_title],
        'bodies':[q_body],
        'titles_masks':[qt_mask],
        'bodies_masks':[qb_mask],
        'similar':[]
        }

        count = 0
        count_pos = 0

        while count < MAX_QCOUNT:
            candidate = None

            if count_pos < MAX_POS and count_pos < len(pos):
                candidate = pos[count_pos]
                sample['similar'].append(count)
                count_pos += 1

            if count < len(rest_qs):
                if rest_qs[count] not in pos:
                    candidate = rest_qs[count]

            if count >= len(rest_qs) or candidate not in self.id2data:
                while True:
                    candidate = random.choice(self.id2data_list)
                    if candidate not in set().union(rest_qs, [query_q]):
                        break

            (title, t_mask), (body, b_mask) = self.id2data[candidate]

            t_mask = pad_mask(t_mask, max_title)
            b_mask = pad_mask(b_mask, max_body)

            pad(title, title_len)
            pad(body, body_len)

            sample['titles'].append(title)
            sample['bodies'].append(body)
            sample['titles_masks'].append(t_mask)
            sample['bodies_masks'].append(b_mask)

            count += 1

        if len(sample['similar']) < MAX_POS:
            while len(sample['similar']) < MAX_POS:
                sample['similar'].append(-1)

        sample['titles'] = torch.LongTensor(sample['titles'])
        sample['bodies'] = torch.LongTensor(sample['bodies'])
        sample['titles_masks'] = torch.LongTensor(sample['titles_masks'])
        sample['bodies_masks'] = torch.LongTensor(sample['bodies_masks'])
        sample['similar'] = torch.LongTensor(sample['similar'])

        return sample


    def createTrainSample(self, q, p, negs, pos, max_title, max_body):
        if q not in self.id2data or p not in self.id2data:
            return None

        (q_title, qt_mask), (q_body, qb_mask) = self.id2data[q]
        (p_title, pt_mask), (p_body, pb_mask) = self.id2data[p]

        qt_mask = pad_mask(qt_mask, max_title)
        qb_mask = pad_mask(qb_mask, max_body)
        pt_mask = pad_mask(pt_mask, max_title)
        pb_mask = pad_mask(pb_mask, max_body)

        pad(q_title, title_len)
        pad(q_body, body_len)
        pad(p_title, title_len)
        pad(p_body, body_len)

        sample = {
        'titles': [q_title, p_title],
        'bodies':[q_body, p_body],
        'titles_masks':[qt_mask, pt_mask],
        'bodies_masks':[qb_mask, pb_mask]
        }

        count_negs = 0

        random.shuffle(negs)

        while count_negs < MAX_QCOUNT-1:

            if count_negs < len(negs):
                neg_candidate = negs[count_negs]
            if count_negs >= len(negs) or neg_candidate not in self.id2data:
                '''
                pick a negative example randomly when a negative example does not exist
                in the dictionary of examples or the data has fewer negative examples than it should
                '''
                while True:
                    neg_candidate = random.choice(self.id2data_list)
                    if neg_candidate not in set().union(negs, pos, [q]):
                        break

            (title, t_mask), (body, b_mask) = self.id2data[neg_candidate]

            t_mask = pad_mask(t_mask, max_title)
            b_mask = pad_mask(b_mask, max_body)

            pad(title, title_len)
            pad(body, body_len)

            sample['titles'].append(title)
            sample['bodies'].append(body)
            sample['titles_masks'].append(t_mask)
            sample['bodies_masks'].append(b_mask)

            count_negs += 1

        sample['titles'] = torch.LongTensor(sample['titles'])
        sample['bodies'] = torch.LongTensor(sample['bodies'])
        sample['titles_masks'] = torch.LongTensor(sample['titles_masks'])
        sample['bodies_masks'] = torch.LongTensor(sample['bodies_masks'])

        return sample  #we do not need y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
