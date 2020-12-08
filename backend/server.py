import os
import asyncio
import websockets
import base64
from threading import Thread
import numpy as np
from mnist_test import mnist_main
from utils import *

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
            # print(n_layers1, features1, drop1, type(n_layers1), type(features1), type(drop1))
            # print(n_layers2, features2, drop2, type(n_layers2), type(features2), type(drop2))
            outputdir, timestamp = make_outputdir()
            train_set, test_set = get_data()
            await websocket.send('Message received. Start training now')
            print(f'Message received. Start training now. Results saved in {timestamp}')
            t1 = Thread(target=mnist_main, args=(n_layers1, features1, drop1, lr1, train_set, test_set, outputdir, 'model1'), daemon=True)
            t2 = Thread(target=mnist_main, args=(n_layers2, features2, drop2, lr2, train_set, test_set, outputdir, 'model2'), daemon=True)
            t1.start()
            t2.start()
            # mnist_main(n_layers1, features1, drop1, lr1, train_set, test_set, outputdir, 'model1')
            # mnist_main(n_layers2, features2, drop2, lr2, train_set, test_set, outputdir, 'model2')

start_server = websockets.serve(hello, "192.168.1.98", 6060)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()