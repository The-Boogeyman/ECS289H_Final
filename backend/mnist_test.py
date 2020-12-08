import os
import time
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class BatchFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, n_layers, output_sizes, drop_out_rate):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                in_features = 1
            else:
                in_features = output_sizes[i - 1]
            if i == n_layers - 1:
                self.layers.extend([
                    nn.Conv2d(in_features, output_sizes[i], 3, 1),
                    nn.ELU(),
                    nn.MaxPool2d(2, 2),
                ])
            else:
                self.layers.extend([
                    nn.Conv2d(in_features, output_sizes[i], 3, 1),
                    nn.ELU(),
                ])
        self.layers.append(BatchFlatten())
        if drop_out_rate > 0.:
            self.layers.extend([
                nn.Dropout(drop_out_rate),
                # nn.Linear(int(((28 - 2 * n_layers) / 2 )**2 * output_sizes[-1] * (1-drop_out_rate)), 10),
                nn.Linear(int(((28 - 2 * n_layers) / 2 )**2 * output_sizes[-1]), 10),
                nn.Softmax(dim=1)
            ])
        else:
            self.layers.extend([
                nn.Linear(int(((28 - 2 * n_layers) / 2 )**2 * output_sizes[-1]), 10),
                nn.Softmax(dim=1)
            ])
        # print(self.layers)
        self.layers = nn.ModuleList(self.layers)
        # self.layer1 = nn.Conv2d(1, 32, 3, 1)
        # self.layer2 = nn.ELU()
        # self.layer3 = nn.MaxPool2d(2, 2)
        # self.layer4 = BatchFlatten()
        # self.layer5 = nn.Linear(32 * 32 * 3, 10)
        # self.layer6 = nn.Softmax()

    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
            # print(x.size())
        # x = self.layers[1](x)
        # print('1: ', x.size())
        # x = self.layers[2](x)
        # print('2: ', x.size())
        # x = self.layers[3](x)
        # print('3: ', x.size())
        # x = self.layers[4](x)
        # print('4: ', x.size())
        # x = self.layers[5](x)
        # print('5: ', x.size())

        # x = self.layer1(x)
        # print('1: ', x.size())
        # x = self.layer2(x)
        # print('2: ', x.size())
        # x = self.layer3(x)
        # print('3: ', x.size())
        # x = self.layer4(x)
        # print('4: ', x.size())
        # x = self.layer5(x)
        # print('5: ', x.size())
        # x = self.layer6(x)
        # print('6: ', x.size())
        return x


def train(model, device, train_loader, optimizer, train_loss_list, epoch, flag):
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    print('{}: Epoch {}, Train: Average loss: {:.4f}'.format(
        flag, epoch, train_loss))


def test(model, device, test_loader, test_loss_list, test_acc_list, epoch, flag):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_accuracy)
    print('{}: Epoch: {}, Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        flag, epoch, test_loss, correct, len(test_loader.dataset), test_accuracy))


def mnist_main(epochs, train_batch_size, lr_step_gamma, n_layers, output_sizes, drop_out_rate, init_lr, train_set, test_set, outputdir, flag):
    assert n_layers == len(
        output_sizes), f'n_layers ({n_layers}) is not equal to len(output_sizes) ({len(output_sizes)})'
    use_cuda = torch.cuda.is_available()
    print(flag, use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    model = Model(n_layers=n_layers, output_sizes=output_sizes,
                  drop_out_rate=drop_out_rate).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=init_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_step_gamma)
    # print(flag, model)
    out_path = os.path.join(outputdir, flag)
    os.makedirs(out_path, exist_ok=True)
    # train and test
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, train_loss_list, epoch, flag)
        test(model, device, test_loader, test_loss_list, test_acc_list, epoch, flag)
        scheduler.step()
        # save trained model & training and testing results
        model_path = os.path.join(out_path, f'epoch.{epoch}.statedict.pt.gz')
        np.save(os.path.join(out_path, f'epoch.{epoch}.train_loss.npy'), train_loss_list)
        np.save(os.path.join(out_path, f'epoch.{epoch}.test_loss.npy'), test_loss_list)
        np.save(os.path.join(out_path, f'epoch.{epoch}.test_acc.npy'), test_acc_list)
        with gzip.open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        # save current version
        cur_model_path = os.path.join(out_path, 'model.statedict.pt.gz')
        np.save(os.path.join(out_path, 'train_loss.npy'), train_loss_list)
        np.save(os.path.join(out_path, 'test_loss.npy'), test_loss_list)
        np.save(os.path.join(out_path, 'test_acc.npy'), test_acc_list)
        with gzip.open(cur_model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        # Below indicates how to load a trained model
        # model_path = os.path.join(out_path, 'model.statedict.pt.gz')
        # with gzip.open(model_path, 'rb') as f:
        #     model = torch.load(f, map_location='cpu')