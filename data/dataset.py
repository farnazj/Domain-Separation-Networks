import data_utils as du
import torch
import gzip
import tqdm
import torch.utils.data as data
import random

NEGATIVE_EXAMPLE_COUNT = 20

def pad(arr, l):
    while len(arr) < l:
            arr.append(0)

class AskUbuntuDataset(data.Dataset):
    def __init__(self, path, id2data, max_title, max_body):
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
                pos = split[1]
                negs = split[2]

                pos = pos.split()
                negs = negs.split()
                negs = [x for x in negs if x not in pos]

                for p in pos:
                    sample = self.createSample(q, p, negs, pos, max_title, max_body)
                    if sample != None:
                        self.dataset.append(sample)


    def createSample(self, q, p, negs, pos, title_len, body_len):
        qarr = []

        if q not in self.id2data or p not in self.id2data:
            return None

        (q_title, qt_mask),(q_body, qb_mask) = self.id2data[q]
        (p_title,pt_mask), (p_body, pb_mask) = self.id2data[p]

        pad(q_title, title_len)
        pad(q_body, body_len)
        pad(p_title, title_len)
        pad(p_body, body_len)

        #q_title = torch.LongTensor(q_title)
        #q_body = torch.LongTensor(q_body)
        #p_title = torch.LongTensor(p_body)
        #p_body = torch.LongTensor(p_body)

        #qarr.append([q_title, q_body])
        #qarr.append([p_title, p_body])

        sample = {'titles': [q_title, p_title], 'bodies':[q_body, p_body], "titles_masks":[qt_mask, pt_mask], "bodies_masks":[qb_mask, pb_mask]}

        count_negs = 0

        random.shuffle(negs)

        while count_negs < NEGATIVE_EXAMPLE_COUNT:

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
            #t_mask = len(title)
            #b_mask = len(body)
            pad(title, title_len)
            pad(body, body_len)
            #qarr.append([title, body])
            sample['titles'].append(title)
            sample['bodies'].append(body)
            sample['titles_masks'].append(t_mask)
            sample['bodies_masks'].append(b_mask)

            count_negs += 1


        #return {'x': qarr, 'y': 1}

        sample['titles'] = torch.LongTensor(sample['titles'])
        sample['bodies'] = torch.LongTensor(sample['bodies'])
        sample['titles_masks'] = torch.LongTensor(sample['titles_masks'])
        sample['bodies_masks'] = torch.LongTensor(sample['bodies_masks'])
        #print sample['titles_masks']
        #print sample['bodies_masks']
        return sample  #we do not need y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
