import argparse, sys, os, torch, datetime, pdb
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import data.data_utils as data_utils
import model.model_utils as model_utils
import train.train_utils as train_utils
import cPickle as pickle


HIDDEN_SIZE = 240
EPOCHS = 42
BATCH_SIZE = 16
WEIGHT_DECAY = [1e-6, 1e-6]
LR = [1e-3, -1e-3]
DROPOUT = 0.1
LAMBDA = 1e-2

TRAIN = False
TEST = False

MODEL = 'lstm'

parser = argparse.ArgumentParser(description='Domain Adaptation in Similar Question Retrieval')
# learning
parser.add_argument('--lr', type=float, nargs=2 ,default=LR, help='initial learning rates for the encoder and the domain discriminator respectively')
parser.add_argument('--hd_size', type=int, default=HIDDEN_SIZE, help='')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size for training [default: 64]')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# model
parser.add_argument('--model_name', nargs="?", type=str, default=MODEL, help="Form of model, i.e dan, rnn, etc.")
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--train', action='store_true', default=TRAIN, help='enable train')
parser.add_argument('--test', action='store_true', default=TEST, help='enable test')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of encoder model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default='model.pt', help='Path where to dump model')
parser.add_argument('--weight_decay', type=float, nargs=2, default=WEIGHT_DECAY, help='weight decays for the encoder and the domain discriminator respectively')
parser.add_argument('--dropout', type=float, default=DROPOUT, help='droput rate')
parser.add_argument('--lambda_d', type=float, default=LAMBDA, help='lambda')



args = parser.parse_args()


if __name__ == '__main__':
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.train:
        train_data, dev_data, embeddings = data_utils.loadDataset(args)
    if args.test:
        test_data = data_utils.loadTest(args)

    # model

    if args.train == True:
        encoder_model, domain_discriminator = model_utils.get_models(embeddings, args)
    elif args.snapshot is None and args.train == False:
        print "Snapshot is None, train flag is False. Must provide snapshot or train the model!"
    else:
        print('\nLoading model from [%s]...' % args.snapshot)

        try:
            model = torch.load(args.snapshot)
        except Exception as e :
            print e
            print("Sorry, This snapshot doesn't exist.")
            exit(1)

    print "encoder model:\n"
    print(encoder_model)


    if args.train:
        print "domain discriminator model:\n"
        print(domain_discriminator)

        train_utils.train_model(train_data, dev_data, encoder_model, domain_discriminator, args)
    if args.test:
        train_utils.test_model(test_data, encoder_model, args)
