import os
import asyncio
import websockets
import base64
import numpy as np
from mnist_test import mnist_main

async def hello(websocket, path):
    await websocket.send('READY')
    stop = False
    while stop == False:
        rec_m = await websocket.recv()
        if 'start' in rec_m:
            _, n_layers1, features1, drop1, lr1, n_layers2, features2, drop2, lr2 = rec_m.split('***')
            n_layers1 = int(n_layers1)
            drop1 = float(drop1)
            lr1 = float(lr1)
            n_layers2 = int(n_layers2)
            drop2 = float(drop2)
            lr2 = float(lr2)
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
            print(n_layers1, features1, drop1, type(n_layers1), type(features1), type(drop1))
            print(n_layers2, features2, drop2, type(n_layers2), type(features2), type(drop2))
            await websocket.send('Message received. Start training now')
            mnist_main(n_layers_1=n_layers1, output_sizes_1=features1, drop_out_rate_1=drop1, init_lr_1=lr1, n_layers_2=n_layers2, output_sizes_2=features2, drop_out_rate_2=drop2, init_lr_2=lr2, epochs=10, train_batch_size=64, lr_step_gamma=0.7)

start_server = websockets.serve(hello, "192.168.1.98", 6060)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()