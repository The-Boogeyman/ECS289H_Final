import os
import gzip
import asyncio
import websockets
import base64
import time
from threading import Thread
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from mnist_test import Model, mnist_main, get_activations
from utils import make_outputdir, get_data


train_set, test_set = get_data()

async def hello(websocket, path):
    await websocket.send('READY')
    stop = False
    while stop == False:
        rec_m = await websocket.recv()
        if 'start' in rec_m:
            _, n_layers1, features1, drop1, lr1, n_layers2, features2, drop2, lr2, epochs, train_batch_size, lr_step_gamma = rec_m.split('***')
            n_layers1 = int(n_layers1)
            drop1 = float(drop1)
            lr1 = float(lr1)
            n_layers2 = int(n_layers2)
            drop2 = float(drop2)
            lr2 = float(lr2)
            epochs = int(epochs)
            train_batch_size = int(train_batch_size)
            lr_step_gamma = float(lr_step_gamma)
            if n_layers1 > 1:
                features1 = features1.split(',')
                features1 = list(map(int, features1))
            else:
                features1 = [int(features1)]
            if n_layers2 > 1:
                features2 = features2.split(',')
                features2 = list(map(int, features2))
            else:
                features2 = [int(features2)]
            # print(n_layers1, features1, drop1, type(n_layers1), type(features1), type(drop1))
            # print(n_layers2, features2, drop2, type(n_layers2), type(features2), type(drop2))
            outputdir, timestamp = make_outputdir()
            await websocket.send('start_training***')
            print(f'Message received. Start training now. Results saved in {timestamp}')
            # TODO
            #  add function to stop training process based on user's selection
            t1 = Thread(target=mnist_main, args=(epochs, train_batch_size, lr_step_gamma, n_layers1, features1, drop1, lr1, train_set, test_set, outputdir, 'model1'), daemon=True)
            t2 = Thread(target=mnist_main, args=(epochs, train_batch_size, lr_step_gamma, n_layers2, features2, drop2, lr2, train_set, test_set, outputdir, 'model2'), daemon=True)
            t1.start()
            t2.start()
        elif 'refresh' in rec_m:
            if os.path.exists(os.path.join(outputdir, 'model1', 'train_loss.npy')) and os.path.exists(os.path.join(outputdir, 'model2', 'train_loss.npy')):
                model1_train_loss = np.load(os.path.join(outputdir, 'model1', 'train_loss.npy')).tolist()
                model1_train_acc = np.load(os.path.join(outputdir, 'model1', 'train_acc.npy')).tolist()
                model1_test_loss = np.load(os.path.join(outputdir, 'model1', 'test_loss.npy')).tolist()
                model1_test_acc = np.load(os.path.join(outputdir, 'model1', 'test_acc.npy')).tolist()
                model2_train_loss = np.load(os.path.join(outputdir, 'model2', 'train_loss.npy')).tolist()
                model2_train_acc = np.load(os.path.join(outputdir, 'model2', 'train_acc.npy')).tolist()
                model2_test_loss = np.load(os.path.join(outputdir, 'model2', 'test_loss.npy')).tolist()
                model2_test_acc = np.load(os.path.join(outputdir, 'model2', 'test_acc.npy')).tolist()
                length = min([len(model1_test_acc), len(model2_test_acc)])
                epoch_list = list(range(1, length + 1))
                sendMsg = f"plotLossTrain***{epoch_list}***{model1_train_loss[0:length]}***{model2_train_loss[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotAccTrain***{epoch_list}***{model1_train_acc[0:length]}***{model2_train_acc[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotLossTest***{epoch_list}***{model1_test_loss[0:length]}***{model2_test_loss[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotAccTest***{epoch_list}***{model1_test_acc[0:length]}***{model2_test_acc[0:length]}"
                await websocket.send(sendMsg)
        elif 'request_img' in rec_m:
            Index = int(rec_m.split('***')[1])
            print('request image: ', Index)
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            test_set_np = test_set.data.numpy()
            test_set_target_np = test_set.targets.numpy()
            sample_data = test_set_np[Index]
            np.save(os.path.join(temp_dir, 'sample_data_org.npy'), sample_data)
            sample_label = test_set_target_np[Index]
            sample_img = Image.fromarray(sample_data)
            sample_img.save(os.path.join(temp_dir, 'sample_img_org.png'))
            with open(os.path.join(temp_dir, 'sample_img_org.png'), 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            sendMsg = "sample_img***" +  img_data + '***' + str(sample_label)
            await websocket.send(sendMsg)
            print('Send the requested original image to the front end')
        elif 'request_activations' in rec_m:
            # TODO
            # There's some issue with this part
            # transform and input to model
            selected_epoch = int(rec_m.split('***')[1])
            print(f'received request to load mode from epoch {selected_epoch}. Start to generate activations')
            model1_path = os.path.join(outputdir, 'model1', f'epoch.{selected_epoch}.pt.gz')
            model2_path = os.path.join(outputdir, 'model2', f'epoch.{selected_epoch}.pt.gz')
            temp_dir = os.path.join(os.getcwd(), 'temp')
            if os.path.exists(model1_path) and os.path.exists(model2_path) and os.path.exists(os.path.join(temp_dir, 'sample_data_org.npy')):
                sample_data = np.load(os.path.join(temp_dir, 'sample_data_org.npy'))
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # model1 = Model(n_layers=n_layers1, output_sizes=features1, drop_out_rate=drop1).to(device)
                # model2 = Model(n_layers=n_layers2, output_sizes=features2, drop_out_rate=drop2).to(device)
                # with gzip.open(model1_path, 'rb') as f1:
                #     # model1.load_state_dict(torch.load(f1))
                #     model1 = torch.load(f1)
                #     model1.to(device)
                # with gzip.open(model2_path, 'rb') as f2:
                #     # model2.load_state_dict(torch.load(f2))
                #     model2 = torch.load(f2)
                #     model2.to(device)
                # transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.1307,), (0.3081,))
                # ])
                # sample_tensor = transform(sample_data).to(device)
                # sample_tensor = sample_tensor.unsqueeze(0)
                # output1 = model1(sample_tensor)
                # output2 = model2(sample_tensor)
                # activations1 = []
                # def hook1(self, input, output):
                #     # print('Inside ' + self.__class__.__name__ + ' forward')
                #     activations1.append(output.detach().squeeze().cpu().numpy())
                # for la in model1.layers:
                #     la.register_forward_hook(hook1)
                # for a in activations1:
                #     print('model1: ', a.shape, type(a))
                # activations2 = []
                # def hook2(self, input, output):
                #     # print('Inside ' + self.__class__.__name__ + ' forward')
                #     activations2.append(output.detach().squeeze().cpu().numpy())
                # for la in model2.layers:
                #     la.register_forward_hook(hook2)
                # for a in activations2:
                #     print('model2: ', a.shape, type(a))
                t3 = Thread(target=get_activations, args=(model1_path, n_layers1, features1, drop1, sample_data, 'model1'), daemon=True)
                t4 = Thread(target=get_activations, args=(model2_path, n_layers2, features2, drop2, sample_data, 'model2'), daemon=True)
                t3.start()
                t4.start()


start_server = websockets.serve(hello, "192.168.1.98", 6060)
# start_server = websockets.serve(hello, "192.168.1.3", 6060)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()