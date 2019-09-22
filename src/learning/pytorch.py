from __future__ import print_function
from builtins import range
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from learning.instance_manager import gen_test_train_set
from preprocessing.FeaturesGenerator import FeaturesGenerationConfig

"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""
import pandas as pd

#build model
class Net(nn.Module):

    def __init__(self, hl):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Inference:

    def __init__(self, input_dir, train_percent, hl, lr):
        torch.manual_seed(1234)
        self.num_epoch = 500
        self.net = Net(hl)
    #choose optimizer and loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def prepare_dataset(self):
        train_df, test_df = gen_test_train_set(self.input_dir, self.train_percent)
        # split x and y (feature and target)
        self.xtrain, self.ytrain = self.process_dataset(train_df)
        self.xtest, self.ytest = self.process_dataset(test_df)

    @staticmethod
    def process_dataset(df):
        df = df.apply(pd.to_numeric)
        datatest_array = df.values
        last_col = len(datatest_array[0]) -1
        xset = datatest_array[:, :last_col]
        yset = datatest_array[:, last_col]
        return xset, yset

    """
    SECTION 2 : Build and Train Model

    Multilayer perceptron model, with one hidden layer.
    input layer : 4 neuron, represents the feature of Iris
    hidden layer : 10 neuron, activation using ReLU
    output layer : 3 neuron, represents the class of Iris

    optimizer = stochastic gradient descent with no batch-size
    loss function = categorical cross entropy
    learning rate = 0.01
    epoch = 500
    """
    def train(self):
        for epoch in range(self.num_epoch):
            X = Variable(torch.Tensor(self.xtrain).float())
            Y = Variable(torch.Tensor(self.ytrain).long())

            # feedforward - backprop
            self.optimizer.zero_grad()
            out = self.net(X)
            loss = self.criterion(out, Y)
            loss.backward()
            self.optimizer.step()

            if (epoch) % 50 == 0:
                print('Epoch [%d/%d] Loss: %.4f'
                      % (epoch + 1, self.num_epoch, loss.item()))

    def test(self):
        # get prediction
        X = Variable(torch.Tensor(self.xtest).float())
        Y = torch.Tensor(self.ytest).long()
        out = self.net(X)
        _, predicted = torch.max(out.data, 1)

        # get accuration
        print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / 30))


def main():
    inference = Inference(input_dir='C:\\research\\falseMedicalClaims\\examples\\model input\\group1',
                         train_percent=0.9, hl=10, hr = 0.1)
    inference.prepare_dataset()
    inference.train()
    inference.test()

if __name__ == '__main__':
    main()




#train

