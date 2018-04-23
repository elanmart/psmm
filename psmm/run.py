import math
import sys
from functools import partial

from torch.nn.utils import clip_grad_norm
import torch as th

from .data import load_ptb, make_batches
from .model import PointerSentinel
from .viz import Visualizer


def detach_all_(*variables):
    """  Detach all Variables inside `variables`.

    Parameters
    ----------
    variables : Tuple or List of Variables
        Variables to detach
    """

    for var in variables:
        var.detach_()

    return variables


def _step(model, x, y, hidden, average=True):
    """  Performs a single step of Pointer-Sentinel model.

    Parameters
    ----------
    model : PointerSentinel
    x : Variable
        input sequence
    y : Variable
        words to predict
    hidden : Variable
        hidden state of the `model`

    Returns
    -------
    hidden : Variable
        new hidden state
    loss : Variable
        loss(model(x), y)
    """

    predictions, hidden = model(x, hidden)
    log_probas = model.mixture_train(*predictions, x, y)

    if average:
        loss = (-log_probas).mean()
    else:
        loss = (-log_probas).sum()

    return hidden, loss


def run_train(data_path, output,
              epochs,
              step, bptt, batch_size,
              lr, clip, lr_decay_step,
              lstm_size, embed_size, n_layers,
              eval_interval,
              use_vis, cuda):
    """ Trains a new model. Run `python main.py --help` for details about the arguments """

    data, word2int     = load_ptb(data_path)
    train, valid, test = data
    nb_steps           = epochs * (train.size(0) //batch_size // step)

    model = PointerSentinel(lstm_size, n_layers, embed_size, vocab_size=len(word2int), k=step)
    model = model.cuda() if cuda else model

    optimizer = th.optim.Adam(params=model.parameters(), lr=lr)

    if use_vis:
        vis = Visualizer(nb_steps=nb_steps, nb_epoch=epochs, update_freq=100)

    _make_batches = partial(make_batches, batch_size=batch_size, step=step, length=bptt, cuda=cuda)

    best_val_loss  = 2**31

    def _train_iter(input, target, hidden):

        model.train()

        optimizer.zero_grad()
        hidden, loss = _step(model, x=input, y=target, hidden=hidden)

        loss.backward()
        clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        loss = float(th.exp(loss))
        assert loss == loss  # nan check

        if use_vis:
            vis.on_batch_done(loss=loss)

        return hidden

    def _run_eval():

        with th.no_grad():

            model.eval()

            if use_vis:
                vis.on_validation_start()

            val_loss = cnt = 0.

            hidden = model.initial_state(batch_size=batch_size)
            for input, target in _make_batches(valid):

                hidden, loss = _step(model, x=input, y=target, hidden=hidden, average=False)
                val_loss += float(loss)
                cnt += input.size(0)

            val_loss = math.exp(val_loss / cnt)

            if use_vis:
                vis.on_validation_done(val_loss)
            else:
                print(val_loss)

            return val_loss

    try:
        for epoch_no in range(epochs):

            if use_vis:
                vis.on_epoch_start()

            hidden = model.initial_state(batch_size=batch_size)
            for j, (input, target) in enumerate(_make_batches(train)):

                hidden = detach_all_(*hidden)
                hidden = _train_iter(input=input, target=target, hidden=hidden)

                if (j + 1) % eval_interval == 0:
                    val_loss = _run_eval()

            if (epoch_no + 1) % lr_decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

    except KeyboardInterrupt:
        print(f"Keyboard interrupt occured, saving model at {output}", file=sys.stderr)

    finally:
        th.save(model, output)
