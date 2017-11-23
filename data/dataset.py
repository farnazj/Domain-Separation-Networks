import data_utils as du
import torch
import gzip
import tqdm
import torch.utils.data as data

def pad(arr, l):
    if len(arr) < l:
        while len(arr) < l:
            arr.append(0)

class AskUbuntuDataset(data.Dataset):
    def __init__(self, path, id2data, max_title, max_body):
        self.path = path
        self.dataset = []
        self.id2data = id2data
        self.title_dim = max_title
        self.body_dim = max_body

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
                    sample = self.createSample(q, p, negs, max_title, max_body)
                    if sample != None:
                        self.dataset.append(sample)


    def createSample(self, q, p, negs, title_len, body_len):
        qarr = []

        if q not in self.id2data or p not in self.id2data:
            return None

        q_title, q_body = self.id2data[q]
        p_title, p_body = self.id2data[p]

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
        sample = {'titles': [q_title, p_title], 'bodies':[q_body, p_body]}

        count_negs = 0
        for np in negs:
            if np not in self.id2data:
                continue
            title, body = self.id2data[np]
            pad(title, title_len)
            pad(body, body_len)
            #qarr.append([title, body])
            sample['titles'].append(title)
            sample['bodies'].append(body)
            count_negs += 1

        if count_negs == 0:
            return None
        #return {'x': qarr, 'y': 1}

        sample['titles'] = torch.LongTensor(sample['titles'])
        sample['bodies'] = torch.LongTensor(sample['bodies'])
        return sample  #we do not need y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
