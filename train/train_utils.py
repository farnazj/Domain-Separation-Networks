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

    for batch in tqdm(data_loader):

        cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        criterion = nn.MultiMarginLoss()
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

        print hidden_rep.size()

        #Calculate cosine similarities here and construct X_scores
        #expected datastructure of hidden_rep = batchsize x number_of_q x hidden_size

        cs_tensor = autograd.Variable(torch.FloatTensor(hidden_rep.size(0), hidden_rep.size(1)-1))

        #calculate cosine similarity for every query vs. neg q pair

        for j in range(1, hidden_rep.size(1)):
            for i in range(hidden_rep.size(0)):
                cs_tensor[i, j-1] = cosine_similarity(hidden_rep[i, 0, ].type(torch.FloatTensor), hidden_rep[i, j, ].type(torch.FloatTensor))
                #print hidden_rep[i, 0, ].type(torch.FloatTensor)
                #print hidden_rep[i, j, ].type(torch.FloatTensor)

        #print cs_tensor

        #X_scores of cosine similarities shold be of size [batch_size, num_questions]
        #y_targets should be all-zero vector of size [batch_size]

        X_scores = torch.stack(cs_tensor, 0)  #??
        #print X_scores
        #print X_scores.size()

        y_targets = autograd.Variable(torch.zeros(hidden_rep.size(0)).type(torch.LongTensor))
        #print y_targets
        #print y_targets.size()

        loss = criterion(X_scores, y_targets)

        print loss

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    #---> Report MAP, MRR, P@1 and P@5

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss
