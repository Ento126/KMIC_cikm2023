import os
import argparse
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from utils import *
from model import KMIC
from data_loader import load_data
from transE.transE_embedding_interface import * 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(222)
torch.manual_seed(222)
 
parser = argparse.ArgumentParser() 
parser.add_argument('--model_name', type=str, default='kgtccli', help='name of model')
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=187, help='the number of epochs')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=0.0001, help='weight of l2 regularization')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--click_sequence_size', type=int, default=256, help='size of user click item sequence')
parser.add_argument('--rnn_input_size', type=int, default=16, help='size of rnn input size, embedding_dim')
parser.add_argument('--rnn_hidden_size', type=int, default=8, help='size of rnn hidden size')
parser.add_argument('--rnn_num_layers', type=int, default=1, help='size of rnn number layers')
parser.add_argument('--lambd', type=float, default=0.5, help='learning rate')
parser.add_argument('--test_start_epoch', type=int, default=35, help='test start epoch')
parser.add_argument('--test_step_len', type=int, default=3, help='test start epoch')
args = parser.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label


def main():    
    data = load_data(args)
    # n_user, n_item, list_users, list_items, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation
    n_user, n_item, list_users, list_items, n_entity, n_relation = data[0], data[1], data[2], data[3], data[4], data[5]
    train_data, eval_data, test_data = data[6], data[7], data[8]
    adj_entity, adj_relation = data[9], data[10]

    df_train_data = numpy2dataframe(train_data)
    df_eval_data = numpy2dataframe(eval_data)
    df_test_data = numpy2dataframe(test_data)

    train_dataset = Dataset(df_train_data)
    eval_dataset = Dataset(df_eval_data)
    test_dataset = Dataset(df_test_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size) 
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size) 
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_entity = torch.from_numpy(adj_entity).to(device=device)
    adj_relation = torch.from_numpy(adj_relation).to(device=device)
    transE_model = load_transE_model(device=device, model= model_transE)

    user_list, train_record, test_record, item_set, k_list, user_click_item_pos = topk_settings(train_data, test_data, n_item)
    model = KMIC(n_user, n_item, list_users, list_items, n_entity, n_relation, adj_entity, adj_relation, user_click_item_pos, args, device, transE_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    epoch = 0
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            model.train()
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            if len(user_ids) == args.batch_size: 
                batch_loss = model.train_forward(user_ids, item_ids, labels) 
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += batch_loss.item()
        train_loss = train_loss / len(train_loader)

        with torch.no_grad():
            test_loss = ctr_eval(model, test_loader, device)
            print('Epoch {} | train_loss {:.4f} test_loss {:.4f}'.format(epoch, train_loss, test_loss))
            if epoch > args.test_start_epoch and epoch % args.test_step_len == 0:
                precision, recall, ndcg = topk_eval(model, user_list, train_record, test_record, item_set, k_list, args.batch_size, device)
                print('precision: @1 @2 @5 @10 @20 \n', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print('recall   : @1 @2 @5 @10 @20 \n', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('ndcg     : @1 @2 @5 @10 @20 \n', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n') 


if __name__ == '__main__':
    main()
