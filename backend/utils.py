import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torchvision import datasets, transforms
from PIL import Image


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data'), train=False,
                              transform=transform)
    return train_set, test_set

def get_total_dataset():
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_data.npy')):
        mnist = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_data.npy'))
        mnist_target = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_target.npy'))
    else:
        train_set, test_set = get_data()
        train_set_np = train_set.data.numpy()
        train_set_target_np = train_set.targets.numpy()
        test_set_np = test_set.data.numpy()
        test_set_target_np = test_set.targets.numpy()
        mnist = np.concatenate((train_set_np, test_set_np), axis=0)
        mnist_target = np.concatenate((train_set_target_np, test_set_target_np), axis=0)
        # print(type(mnist), type(mnist_target))
        # print(mnist.shape, mnist_target.shape)
        np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_data.npy'), mnist)
        np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_target.npy'), mnist_target)
        # example of how to load one image (with index 0-69999)
        # print(mnist[0].shape, mnist_target[0])
        # print(np.min(mnist[0]), np.max(mnist[0]))
        # new_im = Image.fromarray(mnist[0])
        # new_im.save('example.png')
    return mnist, mnist_target


def make_outputdir(postfix=''):
    if postfix is None:
        postfix = ''
    timestamp = time.strftime(f'%Y-%m-%d-%H-%M-%S{postfix}')
    outputdir = os.path.join(os.path.dirname(
        os.path.dirname(os.getcwd())), 'outputs', f'{timestamp}')
    os.makedirs(outputdir, exist_ok=True)
    return outputdir, timestamp

def dim_reduction():
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_coord.npy')):
        embedding = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_coord.npy'))
    else:
        mnist, mnist_target = get_total_dataset()
        mnist.resize((70000, 784))
        embedding = umap.UMAP(random_state=27).fit_transform(mnist)
        np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_coord.npy'), embedding)
        sns.set(style='white', rc={'figure.figsize':(10,8)})
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=mnist_target.astype(int), s=0.1, cmap='Spectral')
        plt.savefig('scatter.png', bbox_inches='tight')
    return embedding

def get_scatterdata():
    _, mnist_target = get_total_dataset()
    embedding = dim_reduction()
    write_path = os.path.join(os.path.dirname(os.getcwd()), 'src', 'data', 'mnistInfo.js')
    with open(write_path, 'a') as ftxt:
        ftxt.write('export default {\n')
        ftxt.write('  mnist: [\n')
    for ll in range(10):
        content_list = []
        for i in range(mnist_target.shape[0]):
            x, y = embedding[i]
            label = mnist_target[i]
            if int(label) == int(ll):
                info = [x, y, label, i]
                content_list.append(info)
        if ll == 9:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    }\n')
                ftxt.write('  ]\n')
                ftxt.write('}\n')
        else:
            with open(write_path, 'a') as ftxt:
                ftxt.write('    {\n')
                ftxt.write(f'      label: \'{ll}\',\n')
                ftxt.write('      content: [\n')
                ftxt.write(f'        {content_list}\n')
                ftxt.write('      ]\n')
                ftxt.write('    },\n')
    print('done!')