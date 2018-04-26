import click

from psmm.run import run_train


@click.group()
def cli():
    pass


@cli.command()
@click.argument("data",        help="Path to the dataset directory (should contain {train,valid,test}.tokens files)")
@click.argument("output",      help="Path to a file where trained model will be saved. WILL OVERRIDE EXISTING FILES")
@click.option("--step",        default=1,     help="Distance between starting indices of consecutive sequences")
@click.option("--bptt",        default=100,   help="Number of timesteps when unrolling the network for bptt")
@click.option("--lr",          default=0.001, help="Learning rate for Adam optimizer")
@click.option("--clip",        default=1,     help="Gradient clipping. If <0, no gradient clipping is performed")
@click.option("--timeout",     default=60,    help="Maximum training time in minutes")
@click.option("--epochs",      default=10,    help="Number of epochs to run")
@click.option("--batch-size",  default=32,    help="Totally not a batch size")
@click.option("--embed-size",  default=64,    help="Size of the word embeddings")
@click.option("--lstm-size",   default=64,    help="Size of the hidden state in LSTM network")
@click.option("--n-layers",    default=2,     help="Number of layers in LSTM network")
@click.option("--cuda",        is_flag=True,  help="Fun fact: CUDA stands for Compute Unified Device Architecture")
def train(data, output, step, bptt, lr, clip, timeout, epochs, batch_size, lstm_size, embed_size, n_layers, cuda):
    run_train(data_path=data, output=output, step=step, bptt=bptt, lr=lr, clip=clip, timeout=timeout, epochs=epochs,
          batch_size=batch_size, lstm_size=lstm_size, embed_size=embed_size, n_layers=n_layers, cuda=cuda)


@cli.command()
@click.argument("model", help="Path to a trained model file")
@click.argument("data",  help="Path to a the dataset directory. Should contains at the 'test.tokens' file")
@click.option("--cuda",  is_flag=True,  help="Fun fact: CUDA stands for Compute Unified Device Architecture")
def evaluate(model, data, cuda):
    raise NotImplementedError()


@cli.command()
@click.argument("model",     help="Path to a trained model file")
@click.argument("data",      help="Path to a file to read.")
@click.option("--bootstrap", default=25,   help="Number of words to read from a document before starting to sample")
@click.option("--cuda",      is_flag=True, help="Fun fact: CUDA stands for Compute Unified Device Architecture")
def sample(model, data, bootstrap, cuda):
    raise NotImplementedError()


if __name__ == '__main__':
    cli()
