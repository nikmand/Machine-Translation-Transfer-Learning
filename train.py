from BCN import BCN
from utils.config import load_config
import os
import argparse
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.nn.functional as F
import torch

from encoder import MTLSTM
from utils.general import number_h

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "model_configs")


def bcn(config, data_file, embeddings, device):
    #   ToDo : fix trainer
    #   extensions : add 2 languages, use a combination of CoVe embeddings (like ELMo)

    inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    labels = data.Field(sequential=False, unk_token=None)

    print('Generating train, dev, test splits')
    # using the IWSLT 2016 TED talk translation task
    # train, dev, test = datasets.IWSLT.splits(root=data_file, exts=['.en', '.de'], fields=[inputs, inputs])
    # using SST
    train, dev, test = datasets.SST.splits(text_field=inputs, label_field=labels, root=data_file)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train, dev, test), batch_size=100, device=torch.device(device) if device >= 0 else None)

    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors(vectors=GloVe(name='840B', dim=300, cache=embeddings))

    labels.build_vocab(train, dev, test)

    model = BCN(config=config, n_vocab=len(inputs.vocab), vocabulary=inputs.vocab.vectors, embeddings=embeddings,
                num_labels=len(labels.vocab.freqs))

    criterion = nn.CrossEntropyLoss()

    if device != -1:
        model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    train_iter.init_epoch()
    print('Generating CoVe')
    for batch_idx, batch in enumerate(train_iter):
        y_pred = model(*batch.text)

        cls_loss = criterion(y_pred, batch.label)

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
