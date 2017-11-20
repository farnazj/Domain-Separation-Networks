import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb


def get_model(embeddings, args):
    print("\nBuilding model...")

    if args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))



class LSTM(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=self.args.hd_size, num_layers=1, batch_first=True, bidirectional=False)
        #self.W_o = nn.Linear(self.args.hd_size,1)


    def forward(self, x_indx):
        input_x = self.embedding_layer(x_indx)
        print input_x
        exit(1)
        h0 = autograd.Variable(torch.zeros(1, self.args.batch_size, self.args.hd_size).type(torch.FloatTensor))
        c0 = autograd.Variable(torch.randn(1, self.args.batch_size, self.args.hd_size).type(torch.FloatTensor))

        output, h_n = self.lstm(input_x, (h0, c0))

        return output
