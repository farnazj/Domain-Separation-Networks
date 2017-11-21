import data_utils as du
import torch.utils.data as data

def pad(arr, l):
    if len(arr) < l:
        while len(arr) < l:
            arr.append(0)

class AskUbuntuDataset(data.Dataset):
    def __init__(self, path, id2data, max_title, max_body):
        self.path = path
        self.dataset = []
        self.id2data = {}
        self.title_dim = max_title
        self.body_dim = max_body

        with gzip.open(path) as gfile:
            for line in tqdm.tqdm(gfile):
                q, pos, negs = line.split('\t')
                pos = pos.split()
                negs = negs.split()
                negs = [x for x in negs if x not in pos]

                for p in pos: #create one sample for every similar question of q
                    sample = self.createSample(q, p, negs, self.title_dim, self.body_dim)
                    self.dataset.append(sample)


    def createSample(q, p, negs, title_len, body_len):
        qarr = []

        q_title, q_body = self.id2data[q]
        p_title, p_body = self.id2data[p]

        pad(q_title, title_len)
        pad(q_body, body_len)
        pad(p_title, title_len)
        pad(p_body, body_len)

        q_title = torch.LongTensor(q_title)
        q_body = torch.LongTensor(q_body)
        p_title = torch.LongTensor(p_body)
        p_body = torch.LongTensor(p_body)

        qarr.append([q_title, q_body])
        qarr.append([p_title, p_body])

        for np in negs:
            title, body = self.id2data[np]
            pad(title, title_len)
            pad(body, body_len)
            qarr.append([title, body])

        #return {'x': qarr, 'y': 1}
        return qarr   #we do not need y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
