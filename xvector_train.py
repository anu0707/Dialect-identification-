################## used Library  ############################################################
import torch
import torch.nn as nn
import os 
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import random
from torch.autograd import Variable
from torch import optim
from models.tdnn import TDNN
from sklearn.metrics import accuracy_score
import h5py
import pickle
from sklearn.preprocessing import OneHotEncoder


lan1id = {'ka':0,'ng':1,'sg':2}
nc = 3 # Number of language classes 
n_epoch = 100 # Number of epochs
IP_dim = 1024 # Dimension of input
##########################################
le=len('/media/data/CygNet_DL2/ananya/layer-analysis/konkani/layer15/train/')

#lan2id={'kar':0,'nor':1,'sou':2,'sin':3} # This is not used here, follow output map instead

####### Function to read data from a file ####
def lstm_data(f):
    #print(f)
    '''hf = h5py.File(f, 'r')
    X = np.array(hf.get('feature'))
    y = np.array(hf.get('target'))
    print(X.shape, "---", y.shape)
    hf.close()'''
    
    X = torch.load(f)
    #print("Y[0]=", y[0])
    #Y1 = y[0]
    #Y1 = np.array([Y1])  
    
    f1 = os.path.splitext(f)[0]     
    lang = f1[le:le+2]  
    #print('lang',lang)
    Y1 = lan1id[lang]    
    Y1 = np.array([Y1]) 
    Y1 = torch.from_numpy(Y1).long()
    
    #Xdata1 = np.array(X)    
    #Xdata1 = torch.from_numpy(X).float() 
    
    #Y1 = torch.from_numpy(Y1).long() 
    return X, Y1  # Return the data and true label


################ X_vector Class #######################

###### TODO: Change context values
class X_vector(nn.Module):
    def __init__(self, input_dim = IP_dim, num_classes=3): # class constructor
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        tdnn1_out = F.relu(self.tdnn1(inputs))
        # print(f'shape of tdnn1 is {tdnn1_out.shape}')
        tdnn2_out = self.tdnn2(tdnn1_out)
        # print(f'shape of tdnn2 is {tdnn2_out.shape}')
        tdnn3_out = self.tdnn3(tdnn2_out)
        # print(f'shape of tdnn3 is {tdnn3_out.shape}')
        tdnn4_out = self.tdnn4(tdnn3_out)
        # print(f'shape of tdnn4 is {tdnn4_out.shape}')
        tdnn5_out = self.tdnn5(tdnn4_out)
        # print(f'shape of tdnn5 is {tdnn5_out.shape}')
        
        ### Stat Pooling        
        mean = torch.mean(tdnn5_out,1)
        # print(f'shape of mean is {mean.shape}')
        std = torch.var(tdnn5_out,1,)
        # print(f'shape of std is {std.shape}')
        stat_pooling = torch.cat((mean,std),1)
        # print(f'shape of stat_pooling is {stat_pooling.shape}')
        segment6_out = self.segment6(stat_pooling)
        
        segment6_out1 = segment6_out[-1]

        # print(f'shape of segment6 is {segment6_out1.shape}')
        #ht = torch.unsqueeze(ht, 0)
        segment6_out1 = torch.unsqueeze(segment6_out1, 0)
        # print(f'shape of segment6 is {segment6_out1.shape}')
        x_vec = self.segment7(segment6_out1)
        # print(x_vec)
        # print(f'shape of x_vec is {x_vec.shape}')
        predictions = self.output(x_vec)
        # print(predictions)
        # print(f'shape of predictions is {predictions.shape}')
        return predictions



######################## X_vector ####################
model = X_vector(IP_dim, nc)
model.cuda()


optimizer =  optim.Adam(model.parameters(), lr=0.0001)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output

#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

files_list=[]
folders = glob.glob('/media/data/CygNet_DL2/ananya/layer-analysis/konkani/layer15/train/*')
print("Folder=", folders)
for folder in folders:
    for f in glob.glob(folder+'/*.pt'):
        files_list.append(f)

print('Total Training files: ',len(files_list))

#files_list=files_list[:100]
l = len(files_list)
txtfl = open('/media/data/CygNet_DL2/ananya/layer-analysis/konkani/train-l15.txt', 'w') # txt file to write training loss and accuracies afte every epoch.


########################
for e in range(n_epoch): # repeat for n_epoch
    i = 0
    cost = 0.0
    random.shuffle(files_list)
    train_loss_list=[]
    full_preds=[]
    full_gts=[]

    for fn in files_list:                          
        XX1, YY1 = lstm_data(fn) # get data from file
        #print("shape of xx1",XX1.shape)
        
        XX1 = torch.unsqueeze(XX1, 1) # Adding one additional dimension at specified position
        #print("shape of xx1",XX1.shape)

        i = i+1  #Counting the number of files

        X1 = np.swapaxes(XX1,0,1)  # changing the axis(similar to transpose)

        # Enable cuda
        X1 = Variable(X1,requires_grad=False).cuda() 
        Y1 = Variable(YY1,requires_grad=False).cuda()

        model.zero_grad() # setting model gradient to zero before calculating gradient
        
        lang_op = model.forward(X1)   # forward propagation the input to the model
        print("lang_op=", lang_op)
        print("lang_op shape=", lang_op.shape)

        T_err = loss_lang(lang_op,Y1)  # loss calculation over the model output with true label
               
        T_err.backward()  # calculating the gradient on loss obtained
        
        optimizer.step() # parameter updation based on gradient calculated in previous step 
        
        train_loss_list.append(T_err.item())

        cost = cost + T_err.item()
            
        print("x-vec-bnf.py: Epoch = ",e+1,"  completed files  "+str(i)+"/"+str(l)+" Loss= %.7f"%(cost/i))
        predictions = np.argmax(lang_op.detach().cpu().numpy(),axis=1) #Convert one hot coded result to label
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)

            ############################

    mean_acc = accuracy_score(full_gts,full_preds)  # accuracy calculation over the true label and predicted label
    mean_loss = np.mean(np.asarray(train_loss_list)) # average loss calculation
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,e+1))      
    path = "/media/data/CygNet_DL2/ananya/layer-analysis/konkani/layer15/model/e_"+str(e+1)+".pth"
    torch.save(model.state_dict(),path) # saving the model parameters 
    txtfl.write(path)
    txtfl.write('\n')
    txtfl.write("acc: "+str(mean_acc))
    txtfl.write('\n')
    txtfl.write("loss: "+str(mean_loss))
    txtfl.write('\n')

###############################################################
