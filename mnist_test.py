import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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
                nn.Linear(int(output_sizes[-1] * output_sizes[-1] * 3 * (1-drop_out_rate)), 10),
                nn.Softmax()    
            ])
        else:
            self.layers.extend([
                nn.Linear(int(output_sizes[-1] * output_sizes[-1] * 3), 10),
                nn.Softmax()
            ])
        # print(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

def backend(n_layers_1, output_sizes_1, drop_out_rate_1, init_lr_1, n_layers_2, output_sizes_2, drop_out_rate_2, init_lr_2, epochs, train_batch_size, lr_step_gamma):
    assert n_layers_1 == len(output_sizes_1), f'n_layers_1 ({n_layers_1}) is not equal to len(output_sizes_1) ({len(output_sizes_1)})'
    assert n_layers_2 == len(output_sizes_2), f'n_layers_2 ({n_layers_2}) is not equal to len(output_sizes_2) ({len(output_sizes_2)})'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_set = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_set_np = train_set.data.numpy()
    train_set_target_np = train_set.targets.numpy()
    test_set_np = test_set.data.numpy()
    test_set_target_np = test_set.targets.numpy()
    print(type(train_set_np), train_set_np.shape, train_set_target_np.shape)
    
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
    model1 = Model(n_layers=n_layers_1, output_sizes=output_sizes_1, drop_out_rate=drop_out_rate_1).to(device)
    model2 = Model(n_layers=n_layers_2, output_sizes=output_sizes_2, drop_out_rate=drop_out_rate_2).to(device)
    optimizer1 = optim.Adadelta(model1.parameters(), lr=init_lr_1)
    optimizer2 = optim.Adadelta(model2.parameters(), lr=init_lr_2)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=lr_step_gamma)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=lr_step_gamma)
    print(model1)
    print(model2)
    # train and save two models
    # for epoch in range(1, epochs + 1):


backend(n_layers_1=2, output_sizes_1=[16, 32], drop_out_rate_1=0.2, init_lr_1=0.01, n_layers_2=3, output_sizes_2=[32, 64, 64], drop_out_rate_2=-1., init_lr_2=0.002, epochs=10, train_batch_size=64, lr_step_gamma=0.7)