""" Utilities to prepare datasets, including tokenization, splitting into batches & more """
import array
import os
from collections import defaultdict

import numpy as np
import torch as th
from torch.autograd import Variable as V


def make_indexer():
    """ Create a defaultdict, for which when key is not present, dict[key] = len(dict)"""
    indexer = defaultdict()
    indexer.default_factory = indexer.__len__

    return indexer


def get(dictionary, key):
    """ Try to get value from a dict `dictionary` for key `key`

    If a standard dict is passed, this behaves as dictionary.get(key). If a defaultdict is passed, this
    function behaves as dictionary[key], returning whatever default_factory produces. """
    try:
        return dictionary[key]
    except KeyError:
        return None


def tokenize(path, word2int=None):
    """  Turn the whole corpus into a single vector of word indices
    
    Parameters
    ----------
    path : str
        Path to read the dataset from.
    word2int : Dict[str, int]
        Dictionary mapping a token to its index. If None, then the dataset will be indexed, and resulting dict returned

    Returns
    -------
    data : LongTensor[:]
        tokenized corpus
    word2int : Dict[str, int]
        Dictionary mapping a token to its index. If it was passed as argument, the same object is returned.
    """

    data     = array.array('l')  # 'l': signed long
    word2int = word2int or make_indexer()

    with open(path) as f:
        for line in f:
            line = line.strip().split()
            line += ['<eos>']

            for token in line:
                idx = get(word2int, token)
                if idx is not None:
                    data.append(idx)

    data = np.frombuffer(data, dtype=np.int64)
    data = th.from_numpy(data)

    return data, dict(word2int)


def batchify(data, batch_size):
    """ Reshape the tokenized corpus so that's its easy to use in mini-batch based training.

    Parameters
    ----------
    data : LongTensor[:]
        Tokenized corpus
    batch_size : int
        Number of sequences in one minibatch

    Returns
    -------
    data : LongTensor[batch_size, :]
    """

    num_batches = data.size(0) // batch_size

    data = data[:num_batches * batch_size]
    data = data.unfold(0, size=num_batches, step=num_batches)

    return data


def make_batches(data, batch_size, step, length, cuda=False, volatile=False):
    """ Generator yielding consecutive minibatches

    Parameters
    ----------
    data : LongTensor[:, :]
        tokenized, 'batchified' dataset
    step : int
        Value that the starting index is incremented by after each yielded batch.
        At `k-th` iteration, we return the sequence starting at index `k * step`
    length : int
        Length of the returned sequence.

    Returns
    -------
    inputs :  Variable[LongTensor[batch_size, :]]
        Context words.
    targets : Variable[LongTensor[batch_size]]
        Target words. Note that this is different than in standard LM, since we provide only one target per sequence.

    """
    if data.dim() == 1:
        data = batchify(data, batch_size=batch_size)
    elif data.size(0) != batch_size:
        raise ValueError("This function prepares batches for language model and "
                         "excpects 1D input, or 2D input of shape (batch_size, None). "
                         "Tensor with shape {} was recieved.".format(data.size()))

    if cuda is True:
        data = data.cuda()

    for idx in range(0, data.size(1)-length, step):
        end     = min(idx+length, data.size(1)-1)
        inputs  = data[:, idx:end]
        targets = data[:, end].squeeze()

        yield V(inputs, volatile=volatile), V(targets, volatile=volatile)


def load_ptb(path):
    """ Load dataset in penn-treebank format from `path`.
    This function expects that 'train.tokens', 'valid.tokens', and 'test.tokens' files can be found inside `path`.

    Parameters
    ----------
    path : str
        path to dataset directory
    """

    train = os.path.join(path, "train.tokens")
    valid = os.path.join(path, "valid.tokens")
    test  = os.path.join(path, "test.tokens")

    assert (os.path.exists(train) and os.path.exists(valid) and os.path.exists(test)), \
        "The data directory you provide should contain {train,valid,test}.tokens files"

    train, word2int = tokenize(train)
    valid, _        = tokenize(valid, word2int=word2int)
    test,  _        = tokenize(test,  word2int=word2int)

    return (train, valid, test), word2int
