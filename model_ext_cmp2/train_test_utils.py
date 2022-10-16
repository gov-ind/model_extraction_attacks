import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random

from utils import to_var
from pdb import set_trace

def load_data():
    test_data = datasets.MNIST(root='../data/', train=False,
        download=True, transform=transforms.ToTensor())


    num_samples = 2000

    rng = np.random.default_rng()
    ix = rng.choice(test_data.data.shape[0], num_samples, replace=False)

    steal_data = test_data.data[ix].float()
    steal_data_targets = test_data.targets[ix]

    mask = np.ones(test_data.data.shape[0], bool)
    mask[ix] = False
    test_data_targets = test_data.targets[mask]
    test_data = test_data.data[mask].float()

    return test_data, test_data_targets, steal_data, steal_data_targets

def load_data2(param):
    hold_out_data = datasets.MNIST(root='../data/', train=True,
        download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='../data/', train=False,
        download=True, transform=transforms.ToTensor())

    indices = list(range(test_dataset.test_data.size(0)))
    split = param['hold_out_size']
    rng = np.random.RandomState()
    rng.shuffle(indices)

    hold_out_idx, test_idx = indices[:split], indices[split:]

    hold_out_sampler = SubsetRandomSampler(hold_out_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    loader_hold_out = torch.utils.data.DataLoader(hold_out_data,
        batch_size=param['hold_out_size'], sampler=hold_out_sampler,
        shuffle=False)
    loader_test = torch.utils.data.DataLoader(test_dataset,
        batch_size=param['test_batch_size'], sampler=test_sampler,
        shuffle=False)

    return loader_hold_out, loader_test

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def test(model, X, y, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, X.shape[0]

    preds = model(X.unsqueeze(1).float())
    _, preds = preds.data.max(1)
    num_correct = (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc

def train(net, X, y, nb_epochs=5, batch_size=64):
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    nb_batches = int(np.ceil(len(X) / float(batch_size)))
    ind = np.arange(len(X))

    # model training
    for epoch in range(nb_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, param['nb_epochs']))
        random.shuffle(ind)

        for m in range(nb_batches):
            i_batch = torch.from_numpy(X[ind[m * batch_size : (m + 1) * batch_size]])
            o_batch = torch.from_numpy(y[ind[m * batch_size : (m + 1) * batch_size]])

            optimizer.zero_grad()

            scores = net(i_batch)
            loss = criterion(scores, o_batch)

            loss.backward()
            optimizer.step()

    return net

def get_preds(model, X):
    scores = model(to_var(torch.from_numpy(X)))
    # Note here that we take the argmax because the adversary
    # only has access to the label (not the probabilities) output
    # by the black-box model
    return np.argmax(scores.data.cpu().numpy(), axis=1)
