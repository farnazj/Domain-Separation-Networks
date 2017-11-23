import os, sys, torch, pdb, datetime
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_model(train_data, dev_data, model, args):


    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print("-------------\nEpoch {}:\n".format(epoch))


        loss = run_epoch(train_data, True, model, optimizer, args)

        print('Train MSE loss: {:.6f}'.format( loss))

        print()

        val_loss = run_epoch(dev_data, False, model, optimizer, args)
        print('Val MSE loss: {:.6f}'.format( val_loss))

        test_loss = run_epoch(test_data, False, model, optimizer, args)
        print('Test MSE loss: {:.6f}'.format( val_loss))

        # Save model
        torch.save(model, args.save_path)

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

    print "here1"
    for batch in tqdm(data_loader):

        print "here2"
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        criterion = nn.MultiMarginLoss(p=1, margin=1, size_average=True)
        #pdb.set_trace()

        if is_training:
            optimizer.zero_grad()

        #out - batch of samples, where every sample is 2d tensor of avg hidden states
        bodies = autograd.Variable(batch['bodies'])
        out_bodies = model(bodies)
        print "body"
        print out_bodies

        titles = autograd.Variable(batch['titles'])
        out_titles = model(titles)
        print "title"
        print out_titles
        hidden_rep = (out_bodies + out_titles)/2

        #Calculate cosine similarities here and construct X_scores
        #expected datastructure of hidden_rep = batchsize x number_of_q x hidden_size

        query_tensor = autograd.Variable(hidden_rep.size(0), hidden_rep.size(2))
        pos_tensor = autograd.Variable(hidden_rep.size(0), hidden_rep.size(2))

        for i in range(hidden_rep.size(0)):
            query_tensor[i] = hidden_rep[i, 0, ]
            pos_tensor[i] = hidden_rep[i, 1, ]

        #cosine similarities query vs pos
        cs = cosine_similarity(query_tensor, pos_tensor)

        cs_tensor = autograd.Variable(hidden_rep.size(0), hidden_rep.size(1))

        for i in range(hidden_rep.size(0)):
            cs_tensor[i][0] = cs[i]

        #calculate cosine similarity for every query vs. neg q pair
        k = 1
        for j in range(2, hidden_rep.size(1)):
            neg_tensor = autograd.Variable(hidden_rep.size(0), hidden_rep.size(2))
            for i in range(hidden_rep.size(0)):
                neg_tensor[i] = hidden_rep[i, j, ]
            cs = cosine_similarity(query_tensor, neg_tensor)
            for c in range(hidden_rep.size(0)):
                cs_tensor[c][k] = cs[i]
            k+=1

        #X_scores of cosine similarities shold be of size [batch_size, num_questions]
        #y_targets should be all-zero vector of size [batch_size]

        X_scores = torch.stack(cs_tensor, 0)  #??
        y_targets = torch.zeros(hidden_rep.size(0)).type(torch.LongTensor)
        loss = criterion(X_scores, y_targets)


        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    #---> Report MAP, MRR, P@1 and P@5

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss
