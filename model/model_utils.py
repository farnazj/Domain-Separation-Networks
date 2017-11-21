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
        return CNN(embeddings, args, feature_size)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))



class LSTM(nn.Module):

    def __init__(self, embeddings, args, feature_size):
        super(LSTM, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=self.args.hd_size, num_layers=1, batch_first=True, bidirectional=False)
        #self.W_o = nn.Linear(self.args.hd_size,1)


    def forward(self, x_index):
        #x_index.view(-1,2,sequence_length, feature_size)
        #x_index.squeeze()
        titles_indices = x_index[:,1:]
        bodies_indices = x_index[:,:1]

        #new_batch_size = self.args.batch_size * sequence_length

        input_x_titles = self.embedding_layer(titles_indices)
        bodies_x_titles = self.embedding_layer(bodies_indices)

        h0 = autograd.Variable(torch.zeros(1, new_batch_size, self.args.hd_size).type(torch.FloatTensor))
        c0 = autograd.Variable(torch.randn(1, new_batch_size, self.args.hd_size).type(torch.FloatTensor))

        output_title, h_n_title = self.lstm(input_x_titles, (h0, c0))
        output_body, h_n_body = self.lstm(input_x_bodies, (h0, c0))

        #reshape the hidden layers
        return (output_title + output_body)/2



class CNN(nn.Module):

    def __init__(self, embeddings, args, feature_size):
        super(CNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.conv1 = nn.Conv1d(feature_size, self.args.hidden_size, kernel_size = 3)

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
        output_titles = F.max_pool1d(F.tanh(self.conv1(input_x_titles)), 3)
        output_bodies = F.max_pool1d(F.tanh(self.conv1(input_x_bodies)), 3)

        #reshape the hidden layers

        return (output_title + output_body)/2
