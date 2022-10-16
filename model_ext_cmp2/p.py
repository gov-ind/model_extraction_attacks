import os
import numpy as np
import random
import torch
import torch.nn as nn

'''seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
'''

import matplotlib.pyplot as plt

from pdb import set_trace

from utils import get_data, get_model, load_model, Net, evaluate, jacobian
from train_test_utils import train
from attacks import random_extract, jacobian_aug_extract, adaptive_extract

X_steal, y_steal, X_test, y_test = get_data(fw='pt')
input_shape = X_steal[0].shape

model = load_model(fw='pt')

print("Original model evaluation:")
#print(evaluate(model, X_test, y_test))

set_trace()

results = [[0.706 , 0.143 , 0.751 , 0.1308, 0.7542],
       [0.7604, 0.118 , 0.7568, 0.1328, 0.7622],
       [0.7974, 0.1626, 0.8094, 0.179 , 0.815 ],
       [0.8286, 0.093 , 0.8418, 0.0996, 0.8514],
       [0.8658, 0.178 , 0.8638, 0.1904, 0.8656],
       [0.8772, 0.1644, 0.8824, 0.1406, 0.8832],
       [0.899 , 0.1378, 0.8956, 0.1292, 0.8962],
       [0.91  , 0.1524, 0.9094, 0.1452, 0.9128],
       [0.9148, 0.1476, 0.9004, 0.1332, 0.9064],
       [0.9158, 0.1536, 0.9116, 0.1742, 0.9156],
       [0.924 , 0.1184, 0.9236, 0.1464, 0.9248]]

df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel("Stealing Dataset Size")
ax.set_ylabel("Stolen Model Accuracy")
for name, group in df.groupby("Method Name"):
    group.plot(1, 2, ax=ax, label=name)
plt.show()

results = []
for len_steal in [100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]: 
    print(f'Len steal: {len_steal}')
    X_steal, y_steal, X_test, y_test = get_data(len_steal, fw='pt')
    
    X_test = X_test[:5000]
    y_test = y_test[:5000]
    X_rand = np.random.randn(*X_steal.shape).astype('f')
    
    accuracies = []

    print('-' * 10 + 'ADAPTIVE' + '-' * 10)
    model_stolen = get_model(fw='pt', input_shape=input_shape)
    acc = adaptive_extract(
        model_stolen,
        model,
        X_steal,
        np.argmax(y_steal, axis=1),
        X_test,
        y_test
    )
    accuracies.append(acc)

    print('-' * 10 + 'RANDOM FAKE' + '-' * 10)
    model_stolen = get_model(fw='pt', input_shape=input_shape)
    acc = random_extract(
        model_stolen,
        model,
        X_rand,
        X_test,
        y_test
    )
    accuracies.append(acc)

    print('-' * 10 + 'RANDOM LEGIT' + '-' * 10)
    model_stolen = get_model(fw='pt', input_shape=input_shape)
    acc = random_extract(model_stolen, model, X_steal, X_test, y_test)
    accuracies.append(acc)
   
    print('-' * 10 + 'JACOBIAN FAKE' + '-' * 10)
    model_stolen = get_model(fw='pt', input_shape=input_shape)
    acc = jacobian_aug_extract(
        model_stolen,
        model,
        X_rand,
        X_test,
        y_test
    )
    accuracies.append(acc)
    
    print('-' * 10 + 'JACOBIAN LEGIT' + '-' * 10)
    model_stolen = get_model(fw='pt', input_shape=input_shape)
    acc = jacobian_aug_extract(
        model_stolen,
        model,
        X_steal,
        X_test,
        y_test
    )
    accuracies.append(acc)

    results.append(accuracies)

set_trace()

strategies = ['Adaptive', 'Random Fake', 'Random Legit', 'Jacobian Fake', 'Jacobian Legit']

import pandas as pd
df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel("Stealing Dataset Size")
ax.set_ylabel("Stolen Model Accuracy")
for name, group in df.groupby("Method Name"):
    group.plot(1, 2, ax=ax, label=name)
plt.show()
