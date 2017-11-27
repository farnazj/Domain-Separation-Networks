import argparse, sys, os, torch, datetime, pdb
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import data.data_utils as data_utils
import model.model_utils as model_utils
import train.train_utils as train_utils
import cPickle as pickle


parser = argparse.ArgumentParser(description='Question Retrieval')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--hd_size', type=int, default=200, help='')
parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=12, help='batch size for training [default: 64]')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# model
parser.add_argument('--model_name', nargs="?", type=str, default='lstm', help="Form of model, i.e dan, rnn, etc.")
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--train', action='store_true', default=True, help='enable train')
parser.add_argument('--test', action='store_true', default=False, help='enable test')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')



args = parser.parse_args()


if __name__ == '__main__':
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    train_data, dev_data, embeddings = data_utils.loadDataset(args)

    # model
    if args.snapshot is None:
        model = model_utils.get_model(embeddings, args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(model)

    if args.train :
        train_utils.train_model(train_data, dev_data, model, args)
    if args.test:
        pass
        #loadtest
        #train_utils.test_model(test_data, model, args)
