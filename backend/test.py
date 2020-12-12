import os
# import gzip
import numpy as np
# from PIL import Image
import torch
# from torchvision import transforms
from utils import get_dataset_from_np, make_outputdir, get_data
from mnist_test import mnist_main

# # Suppose we have already got some trained models. And saved in the folder "2020-12-09-15-19-37/model2"
# # Suppose we want to load model from Epoch 9
# # We need to define the model structure
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
# model = Model(n_layers=2, output_sizes=[32, 64], drop_out_rate=0.5).to(device)
# Epoch = 9
# model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'outputs', '2020-12-09-15-19-37', 'model2', f'epoch.{Epoch}.statedict.pt.gz')

# # This is the correct version of loading a trained model
# with gzip.open(model_path, 'rb') as f:
#     model.load_state_dict(torch.load(f))
#     model.to(device)

# # # Load the dataset. The umap view at front end has saved index information of the below two numpy. So we can use same index to load data
# # # Suppose we want to load data at index 9
# test_set_np = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_test_data.npy'))
# test_set_target_np = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_test_target.npy'))
# Index = 9
# sample_data = test_set_np[Index]
# sample_label = test_set_target_np[Index]

# # print dimension and save the image
# print(type(sample_data), type(sample_label))
# print(sample_data.shape, sample_label)
# im = Image.fromarray(sample_data)
# im.save('sample.png')

# # transform and input to model
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# sample_tensor = transform(sample_data).to(device)
# sample_tensor = sample_tensor.unsqueeze(0)
# output = model(sample_tensor)

# activations = []
# def hook(self, input, output):
#     print('Inside ' + self.__class__.__name__ + ' forward')
#     #print('')
#     #print('input: ', type(input))
#     #print('input[0]: ', type(input[0]))
#     #print('output: ', type(output))
#     #print('')
#     #print('input size:', input[0].size())
#     #print('output size:', output.data.size())
#     #print('output norm:', output.data.norm())
#     activations.append(output.detach().squeeze().cpu().numpy())


# for la in model.layers:
#     la.register_forward_hook(hook)

# model(sample_tensor)

# for a in activations:
#     print(a.shape)

# print("End")

# def threshold(data, t = 15):
#     data[data>=t] = 255
#     data[data<t] = 0
#     return data

# mnist, mnist_target, train, train_target, test, test_target = get_total_dataset()

# # mnist: concatenate train + test
# print(f'mnist total: {type(mnist)}, {type(mnist_target)}, {mnist.shape}, {mnist_target.shape}')
# print(f'train: {type(train)}, {type(train_target)}, {train.shape}, {train_target.shape}')
# print(f'test: {type(test)}, {type(test_target)}, {test.shape}, {test_target.shape}')

# print("Processing to binary images")
# np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_total_data_b.npy'), threshold(mnist))
# np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_train_data_b.npy'), threshold(train))
# np.save(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'mnist_test_data_b.npy'), threshold(test))

train_set, test_set = get_dataset_from_np()
outputdir, timestamp = make_outputdir()
# # print(type(train_set), type(test_set))
mnist_main(30, 512, 0.7, 2, [16, 32], -1., 0.001, train_set, test_set, outputdir, 'model1')