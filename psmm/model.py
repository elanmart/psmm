""" Define the PointerSentinel model class """

import torch as th
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class PointerSentinel(nn.Module):
    """ Implements a Pointer Sentinel Mixture Model of Merity et al. [1]

    Parameters
    ----------
    lstm_size : int
        size of the hidden representation in LSTM layer
    lstm_layers : int
        number of LSTM layers to use
    embedding_size : int
        rank of the embedding matrix
    vocab_size : int
        total number of words in the vocabulary
    k : int
        Number of steps to perform before caching the hidden state.
        The hidden state returned by the network during training is NOT the state after processing the whole input:
            it is the state after processing first `k` elements of the sequence.

    Attributes
    ----------
    embed : nn.Embedding
        Holds embeddings of words. Maps integer indices to d-dimensional vectors.
    lstm : nn.LSTM
        LSTM network. Used to compute representation of a sequence of words
    query : nn.Linear
        Linear projection. Followed by tanh nolinearity maps last hidden vector to query vector
    sentinel : Tensor
        Learned embedding vector for the sentinel, used to compute value of the gating function `g`.
        Size: 1-D (hidden_size, )
    U : Tensor
        Linear projection matrix used to compute logits for the whole vocabulary given last hidden state of LSTM
        Size: 2-D (hidden_size, vocab_size)

    References
    ----------
    .. [1] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models.
    arXiv preprint arXiv:1609.07843.
    """

    def __init__(self, lstm_size, lstm_layers, embedding_size, vocab_size, k):
        super().__init__()

        self.lstm_size = lstm_size
        self.n_layers  = lstm_layers
        self.k         = k

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm  = nn.LSTM(input_size=embedding_size, hidden_size=lstm_size,
                             num_layers=lstm_layers, batch_first=True)

        self.query    = nn.Linear(in_features=lstm_size, out_features=lstm_size)
        self.sentinel = nn.Parameter(th.zeros(1, 1, lstm_size))

        self.U = nn.Parameter(th.zeros(lstm_size, vocab_size))
        self.b = nn.Parameter(th.zeros(vocab_size, ))

        self.reset_parameters()

    def initial_state(self, batch_size):
        """ Returns tuple of zero tensors in the shape expected by the lstm """
        shape = self.n_layers, batch_size, self.lstm_size

        h0 = Variable(self.U.data.new(*shape).zero_())
        c0 = Variable(self.U.data.new(*shape).zero_())

        return h0, c0

    def reset_parameters(self):
        """ Reset parmeters of the mdoel """
        for child in self.children():
            child.reset_parameters()

        self.sentinel.data.normal_(std=0.01)
        self.U.data.normal_(std=0.01)

    def mixture_train(self, ptr_probas, rnn_probas, gates,
                      x, y):
        """ Compute the log-probabilities assigned by the model to
        the target words specified by `y`

        Parameters
        ----------
        ptr_probas : FloatTensor
            see the return values of `forward`
        rnn_probas : FloatTensor
            see the return values of `forward`
        gates : FloatTensor
            see the return values of `forward`
        y : LongTensor
            indices of the target words for each sequence in the batch.
            size : 2-D, (batch-size, 1)
        x : LongTensor
            indices of the input words for each sequence in the batch.
            size : 2-D, (batch-size, seq-length)
        """
        ptr_mask   = (x == y.unsqueeze(1).expand_as(x)).type_as(ptr_probas)
        ptr_scores = (ptr_probas * ptr_mask).sum(1)
        rnn_scores = rnn_probas.gather(dim=1, index=y.unsqueeze(1)).squeeze()
        gates      = gates.squeeze()

        p     = gates * rnn_scores + ptr_scores
        log_p = th.log(p + 1e-12)

        return log_p

    def mixture_sample(self, ptr_probas, rnn_probas, gates, x):
        if x.size(0) != 1:
            raise RuntimeError(f"Sampling is implemented for 'batches' "
                               "of size (1), but {x.size(0)} was found.")

        ptr_probas_expaned = th.zeros_like(rnn_probas)
        ptr_probas_expaned.index_add_(1, x.squeeze(), ptr_probas)

        probas = rnn_probas * gates + ptr_probas_expaned
        return probas

    def forward(self, x, lstm_state):
        """ Compute the distribution over words in the sequence, dist.
        over words in vocabulary, and the gating function

        Parameters
        ----------
        x : LongTensor
            batch of word sequences represented as indices from the vocabulary
            size : 2-D, (batch-size, sequence-len)
        lstm_state : Tuple[FloatTensor]
            initial state of the LSTM (hidden + cell)

        Returns
        -------
        ptr_probas : FloatTensor
            probability mass asigned by the model to the words in the input sequence
            Size: 2-D, (batch-size, sequence-len)
        rnn_probas : FloatTensor
            probability mass asigned by the model to the words in the vocabulary
            Size: 2-D, (batch-size, vocab-size)
        gates : FloatTensor
            values of the gating function
            Size: 1-D, (batch-size, )
        """
        H, sk, hT = self._core(x, lstm_state)

        ptr_probas, gates = self._pointer(H, hT)
        rnn_probas        = self._vanilla_softmax(hT)

        return (ptr_probas, rnn_probas, gates), sk

    def _core(self, x, s0):
        """ Core part of the network, shared between `softmax-rnn` and `pointer` parts

        Parameters
        ----------
        x : LongTensor
            See `forward` method for details.
        s0 : Tuple[FloatTensor]
            See `lstm_state` arg in the `forward`

        Returns
        -------
        H : FloatTensor
            Outputs from the last layer of the LSTM
            Size: (batch-size, seq-len, hid-size)
        sk : Tuple[FloatTensors]
            k-th state (includes cell state) of the all LSTM layers
        hT : FloatTensor
            Last hidden state of the last LSTM layer
            Size: (batch-size, hid-size)
        """
        x = self.embed(x)

        H0, sk = self.lstm(x[:, :self.k, :], s0)
        H1, _  = self.lstm(x[:, self.k:, :], sk)

        H = th.cat((H0, H1), 1)
        hT = H[:, -1, :]

        return H, sk, hT

    def _pointer(self, H, last):
        """ The `pointer` part of the network

        Parameters
        ----------
        H : Tensor
            Hidden representations for each timestep extracted from the last layer of LSTM
            size : (batch-size, sequence-len, hidden-size)
        last : Tensor
            Representations of the whole sequences (Last vectors from H, extracted earlier for efficiency)
            size : (batch-size, hidden-size)
        """
        batch_size, _, hid_size = H.size()

        sentinel = self.sentinel.expand(batch_size, 1, hid_size)
        latents  = th.cat((H, sentinel), 1)  # ::(b, s+1, h)

        query = F.tanh(self.query(last))
        query = query.unsqueeze(2)  # ::(b, h, 1)

        logits  = th.bmm(latents, query).squeeze(2)  # batched-matrix-multiply, ::(b, s+1)
        weigths = F.softmax(logits, dim=1)  # ::(b, s+1)

        probas = weigths[:, :-1]
        gates  = weigths[:, -1].unsqueeze(1)

        return probas, gates

    def _vanilla_softmax(self, last):
        """ The `softmax-rnn` part of the network

        computes distribution over words in vocab given last hidden state
        using a standard linear layer followed by a softmax
        """

        probas = F.softmax(last @ self.U, dim=1)

        return probas