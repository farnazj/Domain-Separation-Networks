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
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=self.args.hd_size, num_layers=1, batch_first=True, bidirectional=False)
        #self.W_o = nn.Linear(self.args.hd_size,1)


    def forward(self, x_index):
        #x_index.view(-1,2,sequence_length, feature_size)
        #x_index.squeeze()
        #titles_indices = x_index[:,1:]
        #bodies_indices = x_index[:,:1]
        #x_index.data.shape[0] -> batch size, x_index.data.shape[1] -> num of questions, x_index.data.shape[2] -> seq length
        reshaped_indices = x_index.view(-1, x_index.data.shape[2])

        new_batch_size = reshaped_indices.data.shape[0]

        embeddings = self.embedding_layer(reshaped_indices)

        h0 = autograd.Variable(torch.zeros(1, new_batch_size, self.args.hd_size).type(torch.FloatTensor))
        c0 = autograd.Variable(torch.randn(1, new_batch_size, self.args.hd_size).type(torch.FloatTensor))

        output, h_n = self.lstm(embeddings, (h0, c0))
        #reshape the hidden layers
        print "1st out"
        print output
        #output = output.view(*(x_index.size() + (self.args.hd_size,)))

        return output



class CNN(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.conv1 = nn.Conv1d(embed_dim, self.args.hidden_size, kernel_size = 3)

    def forward(self, x_index):
        #x_indx.view(-1,2,sequence_length, feature_size)
        #x_indx.squeeze()
        titles_indices = x_index[:,1:]
        bodies_indices = x_index[:,:1]

        input_x_titles = self.embedding_layer(titles_indices)
        bodies_x_titles = self.embedding_layer(bodies_indices)

        #conv receives batch_size * input size* sequence_length (e.g. 16 questions, each having 10 words,
        #each word having an embedding of size sequence_length)

        #the following takes the output of convolutional layers: batch_size * hidden_size * size of output of convolutions
        #to batch_size * hidden_size * 1, may want to squeeze later
        output_titles = F.adaptive_avg_pool1d(F.tanh(self.conv1(input_x_titles)), 1)
        output_bodies = F.adaptive_avg_pool1d(F.tanh(self.conv1(input_x_bodies)), 1)

        #reshape the hidden layers

        return (output_title + output_body)/2
