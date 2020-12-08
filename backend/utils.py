import os
import time
import numpy as np
from torchvision import datasets, transforms

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=False,
                              transform=transform)
    # train_set_np = train_set.data.numpy()
    # train_set_target_np = train_set.targets.numpy()
    # test_set_np = test_set.data.numpy()
    # test_set_target_np = test_set.targets.numpy()
    # print('train: ', train_set_np.shape, train_set_target_np.shape)
    # print('test: ', test_set_np.shape, test_set_target_np.shape)
    return train_set, test_set

def make_outputdir(postfix=''):
    if postfix is None:
        postfix = ''
    timestamp = time.strftime(f'%Y-%m-%d-%H-%M-%S{postfix}')
    outputdir = os.path.join(os.path.dirname(
        os.path.dirname(os.getcwd())), 'outputs', f'{timestamp}')
    os.makedirs(outputdir, exist_ok=True)
    return outputdir, timestamp