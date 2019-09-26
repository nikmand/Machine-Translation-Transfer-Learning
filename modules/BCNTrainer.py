import numpy
import time

import torch
from torch.nn.utils import clip_grad_norm_

from modules.Trainer import Trainer
from utils.logging import epoch_progress
from utils.training import save_checkpoint
from utils.utils import to_device


class BCNTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = None
        self.best_acc = None

    def process_batch(self, inputs, lengths, labels):

        y_pred = self.model(inputs, lengths)

        cls_loss = self.criterion(y_pred, labels)

        return [cls_loss], labels, y_pred

    def train_epoch(self):
        """
        Train the network for one epoch and return the average loss.
        * This will be a pessimistic approximation of the true loss
        of the network, as the loss of the first batches will be higher
        than the true.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.train()
        losses = []

        self.epoch += 1
        epoch_start = time.time()

        if isinstance(self.train_loader, (tuple, list)):
            iterator = zip(*self.train_loader)
        else:
            iterator = self.train_loader

        for i_batch, batch in enumerate(iterator, 1):

            self.step += 1

            for optimizer in self.optimizers:
                optimizer.zero_grad()

            if isinstance(batch.text[0], list):
                X = []
                for item in batch.text[0]:
                    item_array = numpy.array(item)
                    X.append(to_device(torch.from_numpy(item_array),
                                       device=self.device,
                                       dtype=torch.from_numpy(item_array).dtype))

            else:
                X = to_device(batch.text[0], device=self.device, dtype=batch.text[0].dtype)

            y = to_device(batch.label, device=self.device, dtype=torch.long)

            lengths = to_device(batch.text[1], device=self.device, dtype=torch.long)

            batch_loss, _, _ = self.process_batch(X, lengths, y)

            # aggregate the losses into a single loss value
            loss_sum, loss_list = self.return_tensor_and_list(batch_loss)
            losses.append(loss_list)

            # back-propagate
            loss_sum.backward()

            # if self.clip is not None:
            #     for optimizer in self.optimizers:
            #         clip_grad_norm_((p for group in optimizer.param_groups
            #                          for p in group['params']), self.clip)

            # update weights
            for optimizer in self.optimizers:
                optimizer.step()

            if self.step % self.log_interval == 0:
                self.progress_log = epoch_progress(self.epoch, i_batch,
                                                   self.train_batch_size,
                                                   self.train_set_size,
                                                   epoch_start)

            for c in self.batch_end_callbacks:
                if callable(c):
                    c(i_batch, batch_loss)

        return numpy.array(losses).mean(axis=0)

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        labels = []
        posteriors = []
        losses = []
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                if isinstance(batch.text[0], list):
                    X = []
                    for item in batch.text[0]:
                        item_array = numpy.array(item)
                        X.append(to_device(torch.from_numpy(item_array),
                                           device=self.device,
                                           dtype=torch.from_numpy(
                                               item_array).dtype))

                else:
                    X = to_device(batch.text[0], device=self.device, dtype=batch.text[0].dtype)

                y = to_device(batch.label, device=self.device, dtype=torch.long)

                lengths = to_device(batch.text[1], device=self.device,
                                    dtype=torch.long)

                batch_losses, label, cls_logits = self.process_batch(X, lengths, y)
                labels.append(label)
                posteriors.append(cls_logits)

                # aggregate the losses into a single loss value
                loss, _losses = self.return_tensor_and_list(batch_losses)
                losses.append(_losses)

        posteriors = torch.cat(posteriors, dim=0)
        predicted = numpy.argmax(posteriors.cpu(), 1)
        # predicted = predicted.numpy()
        labels_array = numpy.array((torch.cat(labels, dim=0)).cpu())

        return numpy.array(losses).mean(axis=0), labels_array, predicted

    def get_state(self):

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "f1:": self.best_f1,
            "acc": self.best_acc
        }

        return state

    def checkpoint(self, name=None, timestamp=True, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose)

    def test_step(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        labels = []
        posteriors = []
        losses = []
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                if isinstance(batch.text[0], list):
                    X = []
                    for item in batch.text[0]:
                        item_array = numpy.array(item)
                        X.append(to_device(torch.from_numpy(item_array),
                                           device=self.device,
                                           dtype=torch.from_numpy(
                                               item_array).dtype))

                else:
                    X = to_device(batch.text[0], device=self.device, dtype=batch.text[0].dtype)

                y = to_device(batch.label, device=self.device, dtype=torch.long)

                lengths = to_device(batch.text[1], device=self.device,
                                    dtype=torch.long)

                batch_losses, label, cls_logits = self.process_batch(X, lengths, y)
                labels.append(label)
                posteriors.append(cls_logits)

                # aggregate the losses into a single loss value
                loss, _losses = self.return_tensor_and_list(batch_losses)
                losses.append(_losses)

        posteriors = torch.cat(posteriors, dim=0)
        predicted = numpy.argmax(posteriors.cpu(), 1)
        labels_array = numpy.array((torch.cat(labels, dim=0)).cpu())

        return numpy.array(losses).mean(axis=0), labels_array, predicted
