from torch.optim.adam import Adam

from BCN import BCN
from logger.experiment import Experiment
from modules.BCNTrainer import BCNTrainer
from sys_config import EXP_DIR, MODEL_CNF_DIR
from utils.config import load_config
import os
import argparse
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch

from encoder import MTLSTM
from utils.earlystopping import EarlyStopping
from utils.general import number_h
from utils.training import f1_macro, acc, load_checkpoint


def bcn(config, data_file, embeddings, device, chekpoint, dataset, embeddings_type):
    #   extensions : add 2 languages, use a combination of CoVe embeddings (like ELMo)

    inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    labels = data.Field(sequential=False, unk_token=None)

    print('Generating train, dev, test splits')
    if dataset == 'IWSLT':
        # using the IWSLT 2016 TED talk translation task
        train, dev, test = datasets.IWSLT.splits(root=data_file, exts=['.en', '.de'], fields=[inputs, inputs])
    elif dataset == 'SST-2':
        train, dev, test = datasets.SST.splits(text_field=inputs, label_field=labels, root=data_file,
                                               fine_grained=False, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif dataset == 'SST-5':
        train, dev, test = datasets.SST.splits(text_field=inputs, label_field=labels, root=data_file,
                                               fine_grained=True, train_subtrees=True)
    elif dataset == 'IMDB':
        train, test = datasets.IMDB.splits(text_field=inputs, label_field=labels, root=data_file)
        train, dev = train.split(split_ratio=0.9, stratified=True)  # 0.9 in order to be close to the paper
    elif dataset == 'TREC-6':
        train, test = datasets.TREC.splits(text_field=inputs, label_field=labels, root=data_file,
                                           fine_grained=False)
        train, dev = train.split(split_ratio=0.9, stratified=True)
    elif dataset == 'TREC-50':
        train, test = datasets.TREC.splits(text_field=inputs, label_field=labels, root=data_file,
                                           fine_grained=True)
        train, dev = train.split()
    elif dataset == 'SNLI':
        train, dev, test = datasets.SNLI.splits(text_field=inputs, label_field=labels, root=data_file)
    else:
        print('Invalid dataset name detected...')
        return

    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors(vectors=GloVe(name='840B', dim=300, cache=embeddings))

    labels.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=config["train_batch_size"], device=torch.device(device) if device >= 0 else None,
        sort_within_batch=True)

    model = BCN(config=config, n_vocab=len(inputs.vocab), vocabulary=inputs.vocab.vectors, embeddings=embeddings,
                num_labels=len(labels.vocab.freqs), embeddings_type=embeddings_type)

    bcn_params = [p for n, p in model.named_parameters() if "mtlstm" not in n and p.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(bcn_params, lr=0.001)

    if device != -1:
        model.to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in bcn_params
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))
    #####################################
    # Training Pipeline
    #####################################
    trainer = BCNTrainer(model=model, train_loader=None, valid_loader=test_iter, criterion=criterion,
                         device="cpu" if device == -1 else 'cuda',
                         config=config, optimizers=[optimizer])

    state = load_checkpoint(chekpoint)
    model.load_state_dict(state["model"])
    print('Generating CoVe')

    test_loss, y_test, y_pred_test = trainer.test_step()

    print("Test cls loss is {}".format(test_loss))
    print("\n")
    print("F1 on test set is {}".format(f1_macro(y_test, y_pred_test)))
    print("\n")
    print("Accuracy on test set is {}".format(acc(y_test, y_pred_test)))
    print("\n")

    return test_loss, f1_macro(y_test, y_pred_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False,
                        default='basic_model.yaml',
                        help="config file of input data")
    parser.add_argument('--device', default=0, help='Which device to run one; 0 for CPU', type=int)
    parser.add_argument('--data', default='resources', help='where to store data')
    parser.add_argument('--embeddings', default='.embeddings', help='where to store embeddings')
    parser.add_argument('--dataset', default='TREC-6',
                        choices={'IWSLT', 'SST-2', 'SST-5', 'IMDB', 'TREC-6', 'TREC-50', 'SNLI'},
                        help='')
    parser.add_argument('--embeddings_type', default='all',
                        choices={'glove', 'cove_1', 'cove_2', 'all', 'decove'},
                        help='variation of embeddings to be used')
    parser.add_argument('--checkpoint', default='CoVex2_TREC-6')

    args = parser.parse_args()
    input_config = args.input
    data_file = args.data
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    config = load_config(os.path.join(MODEL_CNF_DIR, input_config))
    config["device"] = 'cuda' if args.device >= 0 else 'cpu'
    print("\nThis experiment runs on {}...\n".format(config["device"]))

    bcn(config, data_file, args.embeddings, args.device, args.checkpoint, args.dataset, args.embeddings_type)

if __name__ == '__main__':
    main()
