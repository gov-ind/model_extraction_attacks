import numpy as np
import torch
from utils import jacobian_augmentation, to_var
from train_test_utils import train, test, get_preds
from tqdm.auto import trange
from pdb import set_trace

def random_extract(param, net, X, y, X_test, y_test, file_name_suffix='legit'):
    """
    Train a substitute model using random data
    arXiv:1602.02697
    """

    # Train the substitute and augment dataset alternatively
    acc = -1
    for rho in range(param['data_aug']):
        print("Parent epoch #" + str(rho))
        print("Training data: " + str(len(X)))

        train(net, X, y, param)
        acc = test(net, X_test, y_test)

    torch.save(net.state_dict(),
               param['oracle_name'] + '_random_' + file_name_suffix + '.pkl')
    return acc

def jacobian_sub_extract(param, net, oracle, X, y,
        X_test, y_test, file_name_suffix='legit'):
    """
    Train a substitute model using Jacobian data augmentation
    arXiv:1602.02697
    """

    acc = -1

    # Train the substitute and augment dataset alternatively
    for rho in range(param['data_aug']):
        print("Substitute augmentation  #" + str(rho))
        print("Training data: " + str(len(X)))

        train(net, X, y, param)
        acc = test(net, X_test, y_test)

        # If we are not at last substitute training iteration, augment dataset
        if rho < param['data_aug'] - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X = jacobian_augmentation(net, X, y)

            print("Labeling substitute training data.")
            y = get_preds(oracle, X)

    torch.save(net.state_dict(),
               param['oracle_name'] + '_jac_sub_' + file_name_suffix + '.pkl')
    return acc

def adaptive_extract(x, y, net, oracle, X_test, y_test,
                     param, num_steal=150, verbose=False):
    nb_actions = len(np.unique(y))
    nb_epochs = param['nb_epochs']
    param['nb_epochs'] = 1

    y_avg = np.zeros(nb_actions)

    reward_avg = np.zeros(3)
    reward_var = np.zeros(3)

    # Implement the bandit gradients algorithm
    h_func = np.zeros(nb_actions)
    learning_rate = np.zeros(nb_actions)
    probs = np.ones(nb_actions) / nb_actions
    selected_x = []
    queried_labels = []

    avg_reward = 0.0
    
    for iteration in trange(1, num_steal + 1,
            desc="Knock-off nets adaptive", disable=not verbose):
        #set_trace()
        # Sample an action
        action = np.random.choice(np.arange(0, nb_actions), p=probs)
        
        # Sample data to attack
        sampled_x = _sample_data(x, y, action)
        selected_x.append(sampled_x)

        print(probs)
        #set_trace()
        print(f'{action}')
        # Query the victim classifier
        y_output = get_preds(oracle, np.array([sampled_x]))
        queried_labels.append(y_output[0])

        train(net, np.array([sampled_x]), y_output, param)

        # Test new labels
        y_hat = net(to_var(torch.from_numpy(np.array([
            sampled_x
        ])))).detach().numpy()
        
        # Compute rewards
        reward, reward_avg, reward_var, y_avg = _reward(
            y_output, y_hat, iteration, reward_avg, reward_var, y_avg
        )
        avg_reward = avg_reward + (1.0 / iteration) * (reward - avg_reward)

        # Update learning rate
        learning_rate[action] += 1

        # Update H function
        for i_action in range(nb_actions):
            if i_action != action:
                 h_func[i_action] = (
                    h_func[i_action] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[i_action]
                )
            else:
                h_func[i_action] = h_func[i_action] + 1.0 / learning_rate[action] * (reward - avg_reward) * (
                    1 - probs[i_action]
                )

        # Update probs
        aux_exp = np.exp(h_func)
        probs = aux_exp / np.sum(aux_exp)
 
    param['nb_epochs'] = nb_epochs
    train(net, x, y, param)
    return test(net, X_test, y_test)


def _reward_cert(y_output: np.ndarray) -> float:
    """
    Compute `cert` reward value.

    :param y_output: Output of the victim classifier.
    :return: Reward value.
    """
    largests = np.partition(y_output.flatten(), -2)[-2:]
    reward = largests[1] - largests[0]

    return reward

def _reward_div(y_hat: np.ndarray, n: int, y_avg) -> float:
    """
    Compute `div` reward value.

    :param y_output: Output of the victim classifier.
    :param n: Current iteration.
    :return: Reward value.
    """
    nb_classes = y_hat.shape[1]
    # First update y_avg
    y_avg = y_avg + (1.0 / n) * (y_hat[0] - y_avg)

    # Then compute reward
    reward = 0
    for k in range(nb_classes):
        reward += np.maximum(0, y_hat[0][k] - y_avg[k])

    return reward, y_avg

def _reward_loss(y_output: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute `loss` reward value.

    :param y_output: Output of the victim classifier.
    :param y_hat: Output of the thieved classifier.
    :return: Reward value.
    """
    nb_classes = y_output.shape[1]

    # Compute victim probs
    aux_exp = np.exp(y_output[0])
    probs_output = aux_exp / np.sum(aux_exp)

    # Compute thieved probs
    aux_exp = np.exp(y_hat[0])
    probs_hat = aux_exp / np.sum(aux_exp)

    # Compute reward
    reward = 0
    for k in range(nb_classes):
        reward += -y_output[0][k] * np.log(probs_hat[k])

    return reward

def _reward(y_output: np.ndarray, y_hat: np.ndarray, n: int, reward_avg, reward_var, y_avg) -> np.ndarray:
    """
    Compute `all` reward value.

    :param y_output: Output of the victim classifier.
    :param y_hat: Output of the thieved classifier.
    :param n: Current iteration.
    :return: Reward value.
    """
    set_trace()
    reward_cert = _reward_cert(y_hat)
    reward_div, y_avg = _reward_div(y_hat, n, y_avg)
    label = y_output[0]
    y_output = np.hstack((np.zeros(label),
                          np.array([1]),
                          np.zeros(y_hat.shape[1] - label - 1)))
    y_output = np.expand_dims(y_output, 0)
    reward_loss = _reward_loss(y_output, y_hat)
        
    reward = [reward_cert, reward_div, reward_loss]
    
    reward_avg = reward_avg + (1.0 / n) * (reward - reward_avg)
    reward_var = reward_var + (1.0 / n) * ((reward - reward_avg) ** 2 - reward_var)

    # Normalize rewards
    if n > 1:
        reward = (reward - reward_avg) / np.sqrt(reward_var)
    else:
        reward = [max(min(r, 1), 0) for r in reward]

    return np.mean(reward), reward_avg, reward_var, y_avg

def _sample_data(x: np.ndarray, y: np.ndarray, action: int) -> np.ndarray:
    """
    Sample data with a specific action.

    :param x: An array with the source input to the victim classifier.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
    :param action: The action index returned from the action sampling.
    :return: An array with one input to the victim classifier.
    """
    x_index = x[y == action]
    rnd_idx = np.random.choice(len(x_index))

    return x_index[rnd_idx]

