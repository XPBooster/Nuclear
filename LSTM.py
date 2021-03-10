import re
import pandas as pd
import torch
import torch.nn as nn
import numpy as np;
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn.functional as F
from Optim import Optim
import argparse


class Data(Dataset):
    
    def __init__(self, args, mode, scaler=None):

        self.mode = mode
        self.P = args.window
        self.h = args.horizon
        fin = open(args.data)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.n, self.m = self.rawdat.shape

        self.rawdat = self._split(int(args.train_prob * self.n), int((args.train_prob+args.valid_prob) * self.n))
        self._normalized(scaler)
        self.rawdat = torch.tensor(self.rawdat, dtype=torch.float32, device=args.device)
    
    def _split(self, train_num, valid_num):
        
        if self.mode == 'train':
            return self.rawdat[:train_num, :]
        if self.mode == 'valid':
            return self.rawdat[train_num:valid_num, :]
        if self.mode == 'test':
            return self.rawdat[valid_num:, :]
    
    def _normalized(self, scaler):
    
        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.rawdat = self.scaler.fit_transform(self.rawdat)
        else:
            self.scaler = scaler
            self.rawdat = self.scaler.transform(self.rawdat)
            self.rawdat = self.scaler.transform(self.rawdat)

        
    def __getitem__(self, index):

        start = index
        end = index + self.P

        return self.rawdat[start:end, :-1], self.rawdat[end+1, -1]

    def __len__(self):
        return len(self.rawdat) - self.P - self.h


# for i, data in enumerate(train_loader):
#     print(data.shape)

# valid_set = Data('./data/electricity.txt', 0.6, 0.2, torch.device('cuda:0'), 3, 16, mode='valid', scaler=train_set.scaler)
# valid_loader = DataLoader(dataset=valid_set, batch_size=3, shuffle=True)

# for i, data in enumerate(valid_loader):
#     print(data.shape)

class LSTM(nn.Module):

    def __init__(self, args, feature_num):

        super(LSTM, self).__init__()
        self.P = args.window;
        self.m = feature_num
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.outR = args.outRNN
        self.Ck = args.CNN_kernel;

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); 
        self.GRU1 = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=args.layerRNN, bidirectional=args.Bidirection);
        self.dropout = nn.Dropout(p=args.dropout);
        self.dense = nn.Linear(self.hidR, self.outR)

    def forward(self, x):
        
        #CNN
        c = x.view(-1, 1, self.P, self.m); # (batch,channel=1,p,m=1)
        c = F.relu(self.conv1(c));  # (batch,channel_out,p-kernel_size,1)
        c = self.dropout(c)
        c = torch.squeeze(c, 3); # (batch,c_out,p-kernel_size)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous(); # (time,batch,c_out)
        _, r = self.GRU1(r); # (1, batch, c_out)
        r = torch.squeeze(r, 0)     # (batch, c_out)
        r = self.dense(r)
        r = self.dropout(r)

        return r
    

class LSTM(nn.Module):

    def __init__(self, args, feature_num):

        super(LSTM, self).__init__()
        self.P = args.window;
        self.hidR = args.hidRNN;
        self.outR = args.outRNN
        self.Ck = args.CNN_kernel;
        self.m = feature_num
        self.GRU1 = nn.GRU(input_size=feature_num, hidden_size=self.hidR, num_layers=args.layerRNN, bidirectional=args.Bidirection);
        self.dropout = nn.Dropout(p=args.dropout);
        self.dense = nn.Linear(self.hidR, self.outR)

    def forward(self, x):
        
        #CNN
        c = x.view(-1, self.P, self.m); # (batch,channel=1,p,m=1)
        r = c.permute(1, 0, 2).contiguous(); # (time,batch,c_out)
        _, r = self.GRU1(r); # (1, batch, c_out)
        r = torch.squeeze(r, 0)     # (batch, c_out)
        r = self.dense(r)
        r = self.dropout(r)

        return r
# LSTM
class LSTMModel(BaseEstimator, RegressorMixin):


    def __init__(self, args):

        super(LSTMModel, self).__init__()

        self.train_set = Data(args, mode='train')
        self.valid_set = Data(args, mode='valid', scaler=self.train_set.scaler)
        self.n_epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.loss_hist = []
        self.criterion = nn.L1Loss(size_average=True) if args.L1Loss else nn.MSELoss(size_average=True)
        self.model = LSTM(args, self.train_set.m - 1).to(args.device)  # 注意：只有X预测Y需要这么干！
        self.optimizer = Optim(self.model.parameters(), args.optim, args.lr, args.clip)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')


    def fit(self):

        self.model.train()
        for i in range(self.n_epoch):

            self.loss_hist.append(0)
            train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)
            iter = 0

            for batch_X, batch_y in train_loader:

                self.model.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y.reshape((-1,1)))
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()

                iter+=1

            # self.scheduler.step()

            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]/iter/args.batch_size))

        print('Optimization finished!')

        return self

    # # Test
    # def predict(self, X):
    #     X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu).unsqueeze(1)
    #     self.model.eval()
    #     y = self.scaler_y.inverse_transform(self.model(X).detach().cpu().numpy())

    #     return y

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/0201A13-白城工业园区站.txt', help='location of the data file')
    parser.add_argument('--model', type=str, default='LSTM', help='')
    parser.add_argument('--train_prob', type=float, default=0.7)
    parser.add_argument('--valid_prob', type=float, default=0.15)
    parser.add_argument('--window', type=int, default=1, help='window size')
    parser.add_argument('--horizon', type=int, default=1)

    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=200, help='number of RNN hidden units')
    parser.add_argument('--outRNN', type=int, default=1, help='number of RNN output units')
    parser.add_argument('--layerRNN', type=int, default=1, help='number of RNN layers')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--Bidirection', type=bool, default=False, help='BiGRU')

    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--device', default=torch.device("cuda:0"))
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='log.log', help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    args = parser.parse_args()

    model = LSTMModel(args)
    model.fit()