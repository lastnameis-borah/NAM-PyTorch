""" PARAMETERS """

n_hidden_layers = 1 # number of hidden layers
# hidden_dim = 32 # dimensions of hidden layers
l1 = 0.0 # penalty on output
l2 = 0.1 # weights l2 regularization parameter
dropout = 0.025 # dropout parameter [0,1]
batch_size = 128 # batch_size
epochs = 100 # number of epochs
shallow = True

""" IMPORTS """

import os
import time
import random
import warnings
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader

""" READ & PRE-PROCESS """

df = pd.read_csv('../input/cmsnewsamples/new-smaples.csv').drop(columns = 'Unnamed: 0')
df = df.drop(columns = [i for i in df.columns if '_1' in i])
df['non_hits'] = df[[i for i in df.columns if 'mask' in i]].sum(axis=1)
df = df[df['non_hits']==0].reset_index(drop=True)

df['1/pT'] = df['q/pt'].abs()

features = ['emtf_phi_'+str(i) for i in [0,2,3,4]] + ['emtf_theta_'+str(i) for i in [0,2,3,4]] + ['old_emtf_phi_'+str(i) for i in [0,2,3,4]]

new_features = []
for i in range(len(features)-1):
    for j in range(i+1, (i//4+1)*4):
        new_features.append('delta_'+'_'.join(features[i].split('_')[:-1])+'_'+str((j)%4)+'_'+str(i%4))
        df[new_features[-1]]=df[features[j]]-df[features[i]]

features = new_features[:]

features+=['fr_0', 'fr_2', 'fr_3', 'fr_4']

labels_1 = ['1/pT']

scaler_1 = MinMaxScaler()
df[features] = scaler_1.fit_transform(df[features])

df_features = df[features].copy()

feature_hidden_dim = [min(1000, 2*len(df[features[i]].unique())) for i in range(len(features))]

print(feature_hidden_dim)

df[features+labels_1].head()

""" ExU layer """

class ExU(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = Parameter(torch.Tensor(in_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight = torch.nn.init.normal_(self.weight, mean=3.5, std=0.5)
        self.bias = torch.nn.init.normal_(self.bias, mean=3.5, std=0.5)
        
    def forward(self, inp):
        output = inp-self.bias
        output = output.matmul(torch.exp(self.weight.t()))
        #output = output.matmul(self.weight.t())
        output = F.relu(output)
        
        return output

""" NAM """

class NAM(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_layers, feature_hidden_dim ,dropout = 0, shallow=True):
        super(NAM, self).__init__()
        self.dropout = dropout
        self.model = []
        for i in range(in_dim):
            if n_hidden_layers==0:
                layers = [ExU(1, 1)]
            else:
                layers = [ExU(1, feature_hidden_dim[i]), torch.nn.Dropout(self.dropout)]
                if not shallow:
                    layers+=[torch.nn.Linear(feature_hidden_dim[i], 64), torch.nn.Dropout(self.dropout)]
                    layers+=[torch.nn.Linear(64, 32), torch.nn.Dropout(self.dropout)]
                    layers+=[torch.nn.Linear(32, 1, bias=False)]
                if shallow:
                    layers+=[ExU(feature_hidden_dim[i], 1)]
            self.model.append(torch.nn.Sequential(*layers))
            
        self.model = torch.nn.ModuleList(self.model)

        self.in_dim = in_dim
        self.n_hidden_layers = n_hidden_layers
        self.feature_hidden_dim = feature_hidden_dim
        
        self.summation_params = []
        for i in range(in_dim + 1):
            self.summation_params.append(torch.nn.init.normal_(Parameter(torch.Tensor(1)), mean=0.5, std=0.5))
        self.summation_params = torch.nn.ParameterList(self.summation_params)
            
    def forward(self, x):
        
        output = self.summation_params[0]*self.model[0](x[:,0].reshape(-1,1))
        for i in range(1,self.in_dim):
            output += self.summation_params[i]*self.model[i](x[:,i].reshape(-1,1))
        output += self.summation_params[self.in_dim]
        
#         output = torch.cat([self.summation_params[i]*self.model[i](x[:,i].reshape(-1,1)) for i in range(self.in_dim)], axis=1)
# #         partial_output = output.detach().cpu().numpy()
#         output = output.sum(axis=1)
#         output += self.summation_params[self.in_dim]
    
        return output

""" LOSS FUNCTION """

def criterion(outputs, labels, weights, l1=0):
    loss0 = torch.sqrt(torch.mean((labels-outputs)**2))
    loss1 = torch.sqrt(torch.mean(outputs**2))
    
    return loss0+loss1*l1

""" TRAINING FUNCTION """

def train_nam(model, X_train, Y_train, X_test, Y_test, l1, l2, fold=0, epochs=50, batch_size=128, results_path='./', progress_bar=False):
    
    test_index = list(X_test.index)
    X_val = torch.Tensor(X_train.reset_index(drop=True).iloc[:int(len(X_train)*0.1)].to_numpy())
    Y_val = torch.Tensor(Y_train.reset_index(drop=True).iloc[:int(len(Y_train)*0.1)].to_numpy())
    X_train = torch.Tensor(X_train.reset_index(drop=True).iloc[int(len(X_train)*0.1):].reset_index(drop=True).to_numpy())
    Y_train = torch.Tensor(Y_train.reset_index(drop=True).iloc[int(len(Y_train)*0.1):].reset_index(drop=True).to_numpy())
    X_test = torch.Tensor(X_test.reset_index(drop=True).to_numpy())
    Y_test = torch.Tensor(Y_test.reset_index(drop=True).to_numpy())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers = 4) 
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size) 
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=l2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2, factor=0.2)
    
    l1=torch.tensor(l1)
    
    m_train_loss = []
    m_val_loss = []
    m_test_loss = []
    min_val_loss = float('inf')
    
    for epoch in range(epochs):
      train_loss = 0
      val_loss = 0
      if progress_bar:
          pbar = tqdm(train_loader)
      else:
          pbar = train_loader
      for data in pbar:
        optimizer.zero_grad()
        outputs = model(data[0].to(device))
        labels = data[1].to(device)
        loss = criterion(outputs, labels, model.parameters(), l1)
        l2 = criterion(outputs, labels, model.parameters())
        loss.backward()
        optimizer.step()
        if progress_bar:
          pbar.set_description('Loss: '+str(l2.detach().cpu().numpy()))
        train_loss += l2.detach().cpu().numpy()/len(train_loader)

      for data in val_loader:
        optimizer.zero_grad()
        outputs = model(data[0].to(device))
        labels = data[1].to(device)
        loss = criterion(outputs, labels, model.parameters())
        val_loss += loss.detach().cpu().numpy()/len(val_loader)
      if val_loss<min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'model.pth')
      lr_scheduler.step(val_loss)
      print('Epoch: ', str(epoch+1)+'/'+str(epochs),'| Training Loss: ', train_loss, '| Validation Loss: ', val_loss)
      m_train_loss.append(train_loss)
      m_val_loss.append(val_loss)

    model.load_state_dict(torch.load('model.pth'))
    test_loss = 0
    true = []
    preds = []
    
    for data in test_loader:
      optimizer.zero_grad()
      outputs = model(data[0].to(device))
      labels = data[1].to(device)
      true += list(labels.detach().cpu().numpy().flatten())
      preds += list(outputs.detach().cpu().numpy().flatten())
      loss = criterion(outputs, labels, model.parameters()).detach().cpu().numpy()
      test_loss += loss/len(test_loader)
    
    print('Test Loss: ', test_loss)
    
    OOF_preds = pd.DataFrame()
    OOF_preds['true_value'] = true
    OOF_preds['preds'] = preds
    OOF_preds['row'] = test_index
    OOF_preds.to_csv(os.path.join(results_path, 'OOF_preds_'+str(fold)+'.csv'), index=False)
    
    return m_train_loss, m_val_loss

""" EXPERIMENT """

df = df.sample(frac=1, random_state=242).reset_index(drop=True)

model = NAM(len(features), n_hidden_layers, feature_hidden_dim, dropout)
X_train = df[features].iloc[:int(len(df)*0.8)]
X_test = df[features].iloc[int(len(df)*0.8):]
Y_train = df[labels_1].iloc[:int(len(df)*0.8)]
Y_test = df[labels_1].iloc[int(len(df)*0.8):]

m_train_loss, m_val_loss = train_nam(model, X_train, Y_train, X_test, Y_test, l1, l2, batch_size=batch_size, epochs=epochs)

""" RESULTS """

files = os.listdir('/kaggle/working')
df = pd.concat([pd.read_csv('/kaggle/working/'+i) for i in files if 'OOF_preds_' in i])
df.to_csv('OOF_preds.csv')
df = pd.read_csv('OOF_preds.csv').drop(columns = ['Unnamed: 0'])
df = df.sort_values(by = 'row').reset_index(drop = True)
df['True_pT'] = 1/df['true_value']
df['Predicted_pT'] = 1/df['preds']
df_fcnn = pd.read_csv('../input/1-pt-regression-swiss-activation-new-data/OOF_preds.csv').drop(columns = ['Unnamed: 0'])
df_fcnn = df_fcnn.sort_values(by = 'row').reset_index(drop = True)
df_fcnn['True_pT'] = 1/df_fcnn['true_value']
df_fcnn['Predicted_pT'] = 1/df_fcnn['preds']

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def MAE(df):
    MAE1 = []
    dx = 0.5
    for i in range(int(2/dx),int(150/dx)):
        P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]
        try:
            p = mae(P['True_pT'],P['Predicted_pT'])
        except:
            p=0
        MAE1.append(p)
    MAE1 = MAE1[:196]
    return MAE1
dx = 0.5
MAE1 = MAE(df)
plt.plot([i*dx for i in range(4,200)],MAE1,label = 'NAM')
plt.plot([i*dx for i in range(4,200)],MAE(df_fcnn),label = 'FCNN')
plt.xlabel('pT -->')
plt.ylabel('MAE -->')
plt.legend()
plt.show()

""" PARTIAL DEPENDENCE VISUALIZATION """

class NAM(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_layers, feature_hidden_dim ,dropout = 0, shallow=True):
        super(NAM, self).__init__()
        self.dropout = dropout
        self.model = []
        for i in range(in_dim):
            if n_hidden_layers==0:
                layers = [ExU(1, 1)]
            else:
                layers = [ExU(1, feature_hidden_dim[i]), torch.nn.Dropout(self.dropout)]
                if not shallow:
                    layers+=[torch.nn.Linear(feature_hidden_dim[i], 64), torch.nn.Dropout(self.dropout)]
                    layers+=[torch.nn.Linear(64, 32), torch.nn.Dropout(self.dropout)]
                    layers+=[torch.nn.Linear(32, 1, bias=False)]
                if shallow:
                    layers+=[ExU(feature_hidden_dim[i], 1)]
            self.model.append(torch.nn.Sequential(*layers))
            
        self.model = torch.nn.ModuleList(self.model)

        self.in_dim = in_dim
        self.n_hidden_layers = n_hidden_layers
        self.feature_hidden_dim = feature_hidden_dim
        
        self.summation_params = []
        for i in range(in_dim + 1):
            self.summation_params.append(torch.nn.init.normal_(Parameter(torch.Tensor(1)), mean=0.5, std=0.5))
        self.summation_params = torch.nn.ParameterList(self.summation_params)
            
    def forward(self, x):
        
        output = torch.cat([self.summation_params[i]*self.model[i](x[:,i].reshape(-1,1)) for i in range(self.in_dim)], axis=1)
        partial_output = output.detach().cpu().numpy()
        output = output.sum(axis=1)
        output += self.summation_params[self.in_dim]
        
        return output, partial_output

max_values = df_features.max().to_numpy()
min_values = df_features.min().to_numpy()

model = NAM(len(features), n_hidden_layers, feature_hidden_dim, dropout)
model.load_state_dict(torch.load('model.pth'))
input_to_model = torch.Tensor([[ min_values[j]+(max_values[j]-min_values[j])*i/1000 for i in range(1000)] for j in range(len(features))]).t()

# list(model.parameters())
_, partial_output = model(input_to_model)
partial_output.shape

for i in range(len(features)):
    plt.plot([min_values[i]+(max_values[i]-min_values[i])*j/1000 for j in range(1000)], partial_output[:,i].flatten())
    plt.title('Partial_dependence__of_1/pT_on_'+features[i])
    plt.xlabel(features[i])
    plt.ylabel('Componenet in predicted pT')
    plt.show()

for i in range(len(features)):
    plt.plot([min_values[i]+(max_values[i]-min_values[i])*j/1000 for j in range(1000)], 1/partial_output[:,i].flatten())
    plt.title('Partial_dependence__of_pT_on_'+features[i])
    plt.xlabel(features[i])
    plt.ylabel('Componenet in predicted pT')
    plt.show()