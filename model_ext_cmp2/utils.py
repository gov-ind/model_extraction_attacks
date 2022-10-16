from art.utils import load_mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.auto import trange
from torch.autograd import Variable

from pdb import set_trace

# Load the dataset, and split the test data into test and steal datasets.
(x_train, y_train), (x_test0, y_test0), _, _ = load_mnist()
    
def get_data(len_steal=2000, fw='tf'):
    indices = np.random.permutation(len(x_test0))
    x_steal = x_test0[indices[:len_steal]]
    y_steal = y_test0[indices[:len_steal]]
    x_test = x_test0[indices[len_steal:]]
    y_test = y_test0[indices[len_steal:]]

    if fw == 'pt':
        dims = x_steal.shape[0], x_steal.shape[1], x_steal.shape[2]

        x_steal = x_steal.reshape(x_steal.shape[0], 1, x_steal.shape[1], x_steal.shape[2]).astype('f')
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).astype('f')

    return x_steal, y_steal, x_test, y_test

class FakeNet(nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def get_model(input_shape, fw='tf', num_classes=10, c1=32, c2=64, d1=128):
    if fw == 'tf':
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer, Reshape

        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()

        model = Sequential()
        model.add(Conv2D(c1, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(c2, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(d1, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer="sgd",
                          metrics=['accuracy'])
        return model
    else:
        return FakeNet()

def load_model(fw='tf'):
    if fw == 'tf':
        import keras
        return keras.models.load_model('keras_mnist')
    elif fw == 'pt':
        model = Net()
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
        return model

def evaluate(model, x_test, y_test):
    model.eval()
    preds = torch.argmax(model(torch.from_numpy(x_test)), axis=1)
    return torch.eq(preds, torch.argmax(torch.from_numpy(y_test), axis=1)).sum() / y_test.shape[0]

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives

def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def get_preds(model, X):
    model.eval()
    y = model(torch.from_numpy(X)).detach().numpy()
    return np.argmax(y, axis=1)
