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
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=self.args.hd_size, num_layers=1, batch_first=True, bidirectional=True, dropout=self.args.dropout)
        #self.W_o = nn.Linear(self.args.hd_size,1)


    def forward(self, x_index, masks):

        #x_index.data.shape[0] -> batch size, x_index.data.shape[1] -> num of questions, x_index.data.shape[2] -> seq length
        reshaped_indices = x_index.view(-1, x_index.size(2))

        new_batch_size = reshaped_indices.size(0)

        embeddings = self.embedding_layer(reshaped_indices)

        h0 = autograd.Variable(torch.zeros(2, new_batch_size, self.args.hd_size).type(torch.FloatTensor))
        c0 = autograd.Variable(torch.randn(2, new_batch_size, self.args.hd_size).type(torch.FloatTensor))

        if self.args.cuda:
            h0, c0 = h0.cuda(), c0.cuda()

        output, (h_n, c_n) = self.lstm(embeddings, (h0, c0))

        #seq length hidden state mean pooling (avoiding the padding regions)
        masks_reshaped = masks.view(-1, masks.size(2)).unsqueeze(2).type(torch.FloatTensor)
        masks_expanded = masks_reshaped.expand(masks_reshaped.size(0),masks_reshaped.size(1), output.size(2))

        if self.args.cuda:
            masks_expanded = masks_expanded.cuda()

        masked_seq = masks_expanded * output
        sum_hidden_states = torch.sum(masked_seq, 1)
        true_len = torch.sum(masks_reshaped, 1)

        if self.args.cuda:
            true_len = true_len.cuda()

        averaged_hidden_states = torch.div(sum_hidden_states, true_len)

        #averaged_hidden_states = torch.mean(masked_seq, 1)

        '''
        #last stage pooling:
        idx = (masks - 1).view(-1,1).expand(output.size(0), output.size(2)).unsqueeze(1)
        result = output.gather(1, idx).squeeze()
        '''

        result = averaged_hidden_states.view(x_index.size(0), x_index.size(1),self.args.hd_size * 2)

        return result



class CNN(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim, padding_idx = 0)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False

        self.conv1 = nn.Conv1d(embed_dim, self.args.hd_size, kernel_size = 3, padding = 1)

    def forward(self, x_index, masks):

        reshaped_indices = x_index.view(-1, x_index.size(2))
        new_batch_size = reshaped_indices.size(0)
        embeddings = self.embedding_layer(reshaped_indices)
        #now embeddings is of the form: batchsize * sequence_length * embedding dimension
        #input to conv1d should be of the form batchsize * embedding dimension * sequence_length
        embeddings = embeddings.permute(0,2,1)

        #conv receives batch_size * input size* sequence_length (e.g. 16 questions, each having 10 words,
        #each word having an embedding of size sequence_length)

        #the following takes the output of convolutional layers: batch_size * hidden_size * size of output of convolutions
        #to batch_size * hidden_size * 1
        '''
        idx = (masks - 1).view(-1,1).squeeze(1).data.numpy()
        tangh = F.tanh(convolution)

        averages = []

        for unit_index, unit in enumerate(tangh):
            tangh_result = tangh.narrow(2, 0, idx[unit_index])
        '''
        #mean pooling the convolution layer (avoiding the padding regions)
        convolution = self.conv1(embeddings)

        masks_reshaped = masks.view(-1, masks.size(2)).unsqueeze(1).type(torch.FloatTensor)
        masks_expanded = masks_reshaped.expand(masks_reshaped.size(0), convolution.size(1), masks_reshaped.size(2) )

        if self.args.cuda:
            masks_expanded = masks_expanded.cuda()

        masked_conv = masks_expanded * convolution
        tang = F.tanh(masked_conv)
        sum_hidden_states = torch.sum(tang, 2)
        true_len = torch.sum(masks_reshaped.squeeze(1), 1).unsqueeze(1)

        if self.args.cuda:
            true_len = true_len.cuda()

        averaged_hidden_states = torch.div(sum_hidden_states, true_len)

        '''
        output = F.adaptive_avg_pool1d(tang, 1)

        #lose the dimension of size 1
        output = output.squeeze(2)
        '''
        #reshape back the hidden layers
        result = averaged_hidden_states.view(x_index.size(0), x_index.size(1),self.args.hd_size)

        return result
