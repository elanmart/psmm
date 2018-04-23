import numbers
import time

import numpy as np
import visdom


def as_array(scalar):
    """ Convert scalar to numpy array, so that it can be sent to Visdom server. Can Handle PyTorch tensors / Vars"""
    if type(scalar).__module__.startswith('torch.autograd.variable'):
        scalar = scalar.data
    if type(scalar).__module__.startswith('torch'):
        scalar = scalar.cpu().numpy()
    if isinstance(scalar, numbers.Number):
        scalar = np.array([scalar])

    return scalar


def format_time(t):
    """ Transform seconds into human-readable h:m:s format

    Parameters
    ----------
    t : Union[float, int]
        time to transform [in seconds].
    """
    t = int(t)
    hours,   t = divmod(t, 3600)
    minutes, t = divmod(t, 60)
    seconds    = t

    return f"{hours}h:{minutes}m:{seconds}s"


class Visualizer:
    """ Handles visualization of model training. """

    class TRACES:
        """ Enum for the chart legend """
        TRAIN = 'train ppl'
        VALID = 'valid ppl'

    def __init__(self, nb_steps, nb_epoch, update_freq=1000):
        """
        Parameters
        ----------
        nb_steps : int
            total number of steps the model will perform in one epoch
        nb_epoch : int
            total number of epochs the model will run for
        update_freq : int
            Plots will be updated every `update_freq` steps
        """

        self.nb_steps    = nb_steps
        self.nb_epoch    = nb_epoch
        self.update_freq = update_freq

        self.step_cnt   = 0
        self.epoch_cnt  = 0

        self.train_loss = 0

        self.train_time = 0
        self.valid_time = 0

        self.start     = time.time()
        self.timestamp = time.time()

        self.engine    = visdom.Visdom()
        self.loss_win  = None
        self.text_win  = self.engine.text("")

    def _train_update(self):
        """ Update the panes used to report training progress (train loss and text pane) """
        loss = self.train_loss / self.update_freq
        self.train_loss = 0

        if self.loss_win is None:
            self.loss_win = self.engine.line(X=as_array(self.step_cnt), Y=as_array(loss),
                                             opts=dict(
                                                 legend=[self.TRACES.TRAIN],
                                                 title=f"{self.epoch_cnt}",
                                                 markers=True,
                                                 showlegend=True,
                                             ))
        else:
            self.engine.updateTrace(X=as_array(self.step_cnt), Y=as_array(loss),
                                    win=self.loss_win, name=self.TRACES.TRAIN)

        msg = self._prepare_msg()
        self.engine.text(text=msg, win=self.text_win)

    def _prepare_msg(self):
        """ Prepare the html message reporting about training progress (later displayed inside text pane) """
        bath_per_sec   = self.step_cnt / self.train_time
        total_time     = time.time() - self.start
        time_per_valid = self.valid_time / max(self.epoch_cnt-1, 1)

        progress  = self.step_cnt / self.nb_steps
        ETA_train = self.train_time / progress - self.train_time
        ETA_valid = (self.nb_epoch - self.epoch_cnt + 1) * time_per_valid

        total_time     = format_time(total_time)
        time_per_valid = format_time(time_per_valid)
        ETA_train      = format_time(ETA_train)
        ETA_valid      = format_time(ETA_valid)
        progress       *= 100

        msg = f"""
        <pre>
Running already for:  {total_time}
Epoch:                {self.epoch_cnt} [out of {self.nb_epoch}]
Batches per sec:      {bath_per_sec:.2f}
Avg valid time (min): {time_per_valid}
ETA:                  {ETA_train} train + {ETA_valid} valid
Progress:             {progress:.3f}%
        </pre>
        """

        return msg

    def _valid_update(self, loss):
        """ Updated the chart with a validation loss """
        self.engine.updateTrace(X=as_array(self.step_cnt), Y=as_array(loss),
                                win=self.loss_win, name=self.TRACES.VALID)

    def on_epoch_start(self):
        """ Should be called at the beggining of each epoch. Resets internal timer to estimate remaining train time """
        self.epoch_cnt += 1
        self.timestamp = time.time()

    def on_validation_start(self):
        """ Should be called when validation starts. Starts the timer to estimate validation time. """
        self.timestamp = time.time()

    def on_validation_done(self, loss):
        """  Should be called when validation is finished. Updates Visdom panes.

        Parameters
        ----------
        loss : scalar
            Average loss of the model on the whole validation set
        """

        loss    = as_array(loss)
        elapsed = time.time() - self.timestamp

        self.valid_time += elapsed
        self._valid_update(loss)

    def on_batch_done(self, loss):
        """ Should be called after processing a single mini-batch. Updates internal statistics and perhaps the charts.

        Parameters
        ----------
        loss : scalar
            loss on a single batch
        """

        loss    = as_array(loss)
        elapsed = time.time() - self.timestamp

        self.step_cnt   += 1
        self.train_loss += loss
        self.train_time += elapsed
        self.timestamp  = time.time()

        if self.step_cnt % self.update_freq == 0:
            self._train_update()
