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
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, median_absolute_error, r2_score


class Data(Dataset):
    
    def __init__(self, args, mode, scaler=None):

        self.cuda = args.device
        self.mode = mode
        self.P = args.window
        self.h = args.horizon
        fin = open(args.data)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.n, self.m = self.rawdat.shape

        self.raw_dat = self._split(int(args.train_prob * self.n), int((args.train_prob+args.valid_prob) * self.n))
        self._normalized(scaler);
        self.raw_dat = torch.tensor(self.raw_dat, dtype=torch.float32, device=args.device)
        
#         tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
            
#         if self.cuda:
#             self.scale = self.scale.cuda();
#         self.scale = Variable(self.scale);
        
#         self.rse = normal_std(tmp);
#         self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));
    
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
            self.scaler.fit(self.raw_dat)
        else:
            self.scaler = scaler
        if self.mode == 'train':
            self.raw_dat = self.scaler.transform(self.raw_dat)
        if self.mode == 'valid':
            self.raw_dat = self.scaler.transform(self.raw_dat)
        if self.mode == 'test':
            self.raw_dat = self.scaler.transform(self.raw_dat)

        
    def __getitem__(self, index):

        start = index
        end = index + self.P

        return self.raw_dat[start:end], self.raw_dat[end+1]

    def __len__(self):
        return len(self.raw_dat) - self.P - self.h



# for i, data in enumerate(train_loader):
#     print(data.shape)

# valid_set = Data('./data/electricity.txt', 0.6, 0.2, torch.device('cuda:0'), 3, 16, mode='valid', scaler=train_set.scaler)
# valid_loader = DataLoader(dataset=valid_set, batch_size=3, shuffle=True)

# for i, data in enumerate(valid_loader):
#     print(data.shape)

class LSTNet(nn.Module):

    def __init__(self, args, feature_num):

        super(LSTNet, self).__init__()
        self.P = args.window
        self.m = feature_num
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)) 
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
 
    def forward(self, x):

        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.P, self.m) # (batch,1,p,m)
        c = F.relu(self.conv1(c))  # (batch,c_out,p-kernel_size,1)
        c = self.dropout(c)
        c = torch.squeeze(c, 3) # (batch,c_out,p-kernel_size)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous() # (time,batch,c_out)
        _, r = self.GRU1(r) # (1, batch, feature)
        r = self.dropout(torch.squeeze(r,0)) # (batch, c_out)

        
        #skip-rnn 
        
        if (self.skip > 0):
            
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res
    

# LSTM
class LSTNetModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, args):
        super(LSTNetModel, self).__init__()

        # Parameter Assignment
        self.train_set = Data(args, mode='train')
        self.valid_set = Data(args, mode='valid', scaler=self.train_set.scaler)
        self.epoch = 0
        self.n_epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.input_dim = self.train_set.raw_dat.shape[1]
        self.criterion = nn.L1Loss(size_average=False) if args.L1Loss else nn.MSELoss(size_average=False)
        self.model = LSTNet(args, self.input_dim).to(args.device)
        self.optimizer = Optim(self.model.parameters(), args.optim, args.lr, args.clip)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')
        
        self.eval = dict()
        self.eval['loss'] = []
        self.eval['mean_absolute_error'] = []
        self.eval['explained_variance_score'] = []
        self.eval['mean_squared_error'] = []
        self.eval['median_absolute_error'] = []
        self.eval['r2_score'] = []

    def fit(self):

        self.model.train()
        for i in range(self.n_epoch):

            self.eval['loss'].append(0)
            train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)
            
            idx = 0
            for batch_X, batch_y in train_loader:

                self.model.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y) # shape
                self.eval['loss'][-1] += loss.item()
                loss.backward()
                self.optimizer.step()
                idx += 1

            # self.scheduler.step()
            self.epoch += 1
            print('Epoch: {}, Loss: {}'.format(self.epoch + 1, self.eval['loss'][-1]/idx/args.batch_size/self.input_dim))
            
        print('Optimization finished!')


    def valid(self):

        with torch.no_grad():
            self.model.eval()

            batchy_list = torch.tensor([], dtype=torch.float32, device=args.device)
            output_list = torch.tensor([], dtype=torch.float32, device=args.device)
            valid_loader = DataLoader(dataset=self.valid_set, batch_size=self.batch_size, shuffle=True)
            loss_valid = 0
            idx = 0

            for batch_X, batch_y in valid_loader:

                self.model.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y) # shape
                loss_valid += loss.item()

                batchy_list = torch.cat((batchy_list, batch_y), axis=0)
                output_list = torch.cat((output_list, output), axis=0)
                idx += 1

            batchy_list = batchy_list.cpu()
            output_list = output_list.cpu()
            self.eval['mean_squared_error'].append(mean_squared_error(batchy_list,output_list, multioutput='raw_values'))
            self.eval['mean_absolute_error'].append(mean_absolute_error(batchy_list, output_list, multioutput='raw_values'))
            self.eval['median_absolute_error'].append(median_absolute_error(batchy_list, output_list, multioutput='raw_values'))
            self.eval['explained_variance_score'].append(explained_variance_score(batchy_list, output_list, multioutput='raw_values'))
            self.eval['r2_score'].append(r2_score(batchy_list, output_list, multioutput='raw_values'))

        print('Epoch: {}, Valid Loss: {}'.format(self.epoch, np.array(self.eval['mean_squared_error']).mean(axis=1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/test.txt', help='location of the data file')
    # parser.add_argument('--data', type=str, default='./data/traffic.txt',
    #                     help='location of the data file')
    parser.add_argument('--model', type=str, default='LSTNet', help='')
    parser.add_argument('--train_prob', type=float, default=0.7)
    parser.add_argument('--valid_prob', type=float, default=0.15)
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=24 * 7, help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=6,)
    parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--dropout', type=float, default=0.1, help='(0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321,)
    parser.add_argument('--device', default=torch.device("cuda:0"))
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='log.log', help='path to save the results')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--skip', type=float, default=24)
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    args = parser.parse_args()

    model = LSTNetModel(args)
    model.fit()
    model.valid()
    np.savetxt('loss.txt', model.eval['loss'])
    np.savetxt('r2.txt', model.eval['r2_score'])