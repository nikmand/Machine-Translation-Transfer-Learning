from BCN import BCN
from utils.config import load_config
import os
import argparse
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch

from encoder import MTLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "model_configs")


def bcn(config, data_file, embeddings, device):
    inputs = data.Field(lower=True, include_lengths=True, batch_first=True)

    print('Generating train, dev, test splits')
    # using the IWSLT 2016 TED talk translation task
    train, dev, test = datasets.IWSLT.splits(root=data_file, exts=['.en', '.de'], fields=[inputs, inputs])
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train, dev, test), batch_size=100, device=torch.device(device) if device >= 0 else None)

    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors(vectors=GloVe(name='840B', dim=300, cache=embeddings))

    model = BCN(config=config, n_vocab=len(inputs.vocab), vocabulary=inputs.vocab.vectors, embeddings=embeddings)

    if device != -1:
        model.to(device)
    print(model)

    train_iter.init_epoch()
    print('Generating CoVe')
    for batch_idx, batch in enumerate(train_iter):
        if batch_idx > 0:
            break
        model(*batch.src)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False,
                        default='basic_model.yaml',
                        help="config file of input data")
    parser.add_argument('--device', default=-1, help='Which device to run one; -1 for CPU', type=int)
    parser.add_argument('--data', default='resources', help='where to store data')
    parser.add_argument('--embeddings', default='.embeddings', help='where to store embeddings')

    args = parser.parse_args()
    input_config = args.input
    data_file = args.data
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print("\nThis experiment runs on gpu {}...\n".format(str(args.device)))

    config = load_config(os.path.join(MODEL_CNF_DIR, input_config))
    config["gpu"] = args.device

    bcn(config, data_file, args.embeddings, args.device)


if __name__ == '__main__':
    main()
