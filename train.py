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
from utils.training import f1_macro, acc


def bcn(config, data_file, embeddings, device):
    #   extensions : add 2 languages, use a combination of CoVe embeddings (like ELMo)

    name = "test_model"

    inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    labels = data.Field(sequential=False, unk_token=None)

    print('Generating train, dev, test splits')
    # using the IWSLT 2016 TED talk translation task
    # train, dev, test = datasets.IWSLT.splits(root=data_file, exts=['.en', '.de'], fields=[inputs, inputs])
    # using SST
    train, dev, test = datasets.SST.splits(text_field=inputs, label_field=labels, root=data_file, fine_grained=False,
                                           train_subtrees=True,
                                           filter_pred=lambda ex: ex.label != 'neutral')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train, dev, test), batch_size=config["train_batch_size"], device=torch.device(device) if device >= 0 else None)

    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors(vectors=GloVe(name='840B', dim=300, cache=embeddings))

    labels.build_vocab(train, dev, test)

    model = BCN(config=config, n_vocab=len(inputs.vocab), vocabulary=inputs.vocab.vectors, embeddings=embeddings,
                num_labels=len(labels.vocab.freqs))

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
    trainer = BCNTrainer(model=model, train_loader=train_iter, valid_loader=dev_iter, criterion=criterion, device="cpu",
                         config=config, optimizers=optimizer)

    print('Generating CoVe')

    ####################################################################
    # Experiment: logging and visualizing the training process
    ####################################################################
    exp = Experiment(name, config, src_dirs=None, output_dir=EXP_DIR)
    exp.add_metric("ep_loss", "line", "epoch loss class", ["TRAIN", "VAL"])
    exp.add_metric("ep_f1", "line", "epoch f1", ["TRAIN", "VAL"])
    exp.add_metric("ep_acc", "line", "epoch accuracy", ["TRAIN", "VAL"])

    exp.add_value("epoch", title="epoch summary")
    exp.add_value("progress", title="training progress")

    ####################################################################
    # Training Loop
    ####################################################################
    best_loss = None
    early_stopping = EarlyStopping("min", config["patience"])

    for epoch in range(1, config["epochs"] + 1):
        train_loss = trainer.train_epoch()
        val_loss, y, y_pred = trainer.eval_epoch()

        # Calculate accuracy and f1-macro on the evaluation set
        exp.update_metric("ep_loss", train_loss.item(), "TRAIN")
        exp.update_metric("ep_loss", val_loss.item(), "VAL")
        exp.update_metric("ep_f1", 0, "TRAIN")
        exp.update_metric("ep_f1", f1_macro(y, y_pred), "VAL")
        exp.update_metric("ep_acc", 0, "TRAIN")
        exp.update_metric("ep_acc", acc(y, y_pred), "VAL")

        print()
        epoch_log = exp.log_metrics(["ep_loss", "ep_f1", "ep_acc"])
        print(epoch_log)
        exp.update_value("epoch", epoch_log)

        # Save the model if the val loss is the best we've seen so far.
        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            trainer.best_acc = acc(y, y_pred)
            trainer.best_f1 = f1_macro(y, y_pred)
            trainer.checkpoint(name=name)

        if early_stopping.stop(val_loss):
            print("Early Stopping (according to cls loss)....")
            break

        print("\n" * 2)

    return best_loss, trainer.best_acc, trainer.best_f1


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
