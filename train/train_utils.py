import os, sys, torch, pdb, datetime
from operator import itemgetter
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def updateScores(args, cs_tensor, similar, i, sum_av_prec, sum_ranks, num_samples, top_5, top_1):
    scores_list = []
    for j in range(20):
        x = cs_tensor[i, j].data
        x = x.numpy().item()
        scores_list.append( (x, j) )

    scores_list = sorted(scores_list, reverse = True, key=itemgetter(0))

    count = 0.0
    last_index = -1
    sum_prec = 0.0
    similar_indices = []
    flag = 0

    for k in similar:
        if k != -1:
            similar_indices.append(k)

    for j in range(20):
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
        sum_prec /= last_index

    sum_av_prec += sum_prec
    num_samples += 1

    return sum_av_prec, sum_ranks, num_samples, top_5, top_1

def train_model(train_data, dev_data, model, args):
    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr, weight_decay=args.weight_decay)

    if args.train:
        model.train()

    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        run_epoch(train_data, True, model, optimizer, args)

        torch.save(model, args.save_path)

        print "*******dev********"
        run_epoch(dev_data, False, model, optimizer, args)



def test_model(test_data, model, args):
    if args.cuda:
        model = model.cuda()

    print "*******test********"
    run_epoch(test_data, False, model, None, args)


def run_epoch(data, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    losses = []

    if is_training:
        model.train()
    else:
        model.eval()

    sum_av_prec = 0.0
    sum_ranks = 0.0
    num_samples = 0.0
    top_5 = 0.0
    top_1 = 0.0

    for batch in tqdm(data_loader):

        cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        criterion = nn.MultiMarginLoss(margin=0.4)
        #pdb.set_trace()

        if is_training:
            optimizer.zero_grad()

        #out - batch of samples, where every sample is 2d tensor of avg hidden states
        bodies = autograd.Variable(batch['bodies'])
        bodies_masks = autograd.Variable(batch['bodies_masks'])
        out_bodies = model(bodies, bodies_masks)

        titles = autograd.Variable(batch['titles'])
        titles_masks = autograd.Variable(batch['titles_masks'])
        out_titles = model(titles, titles_masks)

        hidden_rep = (out_bodies + out_titles)/2

        #Calculate cosine similarities here and construct X_scores
        #expected datastructure of hidden_rep = batchsize x number_of_q x hidden_size

        cs_tensor = autograd.Variable(torch.FloatTensor(hidden_rep.size(0), hidden_rep.size(1)-1))

        #calculate cosine similarity for every query vs. neg q pair

        for j in range(1, hidden_rep.size(1)):
            for i in range(hidden_rep.size(0)):
                cs_tensor[i, j-1] = cosine_similarity(hidden_rep[i, 0, ], hidden_rep[i, j, ])
                #cs_tensor[i, j-1] = cosine_similarity(hidden_rep[i, 0, ].type(torch.FloatTensor), hidden_rep[i, j, ].type(torch.FloatTensor))

        X_scores = torch.stack(cs_tensor, 0)
        y_targets = autograd.Variable(torch.zeros(hidden_rep.size(0)).type(torch.LongTensor))


        if is_training:
            loss = criterion(X_scores, y_targets)
            print "Loss in batch", loss.data

            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().data[0])

        else:
            #Average Precision = (sum_{i in j} P@i / j)  where j is the last index
            for i in range(args.batch_size):
                sum_av_prec, sum_ranks, num_samples, top_5, top_1 = \
                updateScores(args, cs_tensor, batch['similar'][i], i, sum_av_prec, sum_ranks, num_samples, top_5, top_1)

    # Calculate epoch level scores
    if is_training:
        avg_loss = np.mean(losses)
        print('Average Train loss: {:.6f}'.format(avg_loss))
        print()
    else:
        _map = sum_av_prec/num_samples
        _mrr = sum_ranks/num_samples
        _pat5 = top_5/(num_samples*5)
        _pat1 = top_1/num_samples
        print('MAP: {:.3f}'.format(_map))
        print('MRR: {:.3f}'.format(_mrr))
        print('P@1: {:.3f}'.format(_pat1))
        print('P@5: {:.3f}'.format(_pat5))
