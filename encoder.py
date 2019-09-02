import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo


WMT_LSTM = 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-8f474287.pth'
MODEL_CACHE = '.embeddings'


class MTLSTM(nn.Module):

    def  __init__(self, n_vocab=None, vectors=None, residual_embeddings=False, model_cache=MODEL_CACHE):
        """
        Arguments:
            n_vocab (int): If not None, initialize MTLSTM with an embedding matrix with n_vocab vectors
            vectors (Float Tensor): If not None, initialize embedding matrix with specified vectors (These should be 300d CommonCrawl GloVe vectors)
            residual_embedding (bool): If True, concatenate the input GloVe embeddings with contextualized word vectors as final output
            model_cache (str): path to the model file for the MTLSTM to load pretrained weights (defaults to the best MTLSTM from (McCann et al. 2017) --
                               that MTLSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.
        """

        super(MTLSTM, self).__init__()
        self.residual_embeddings = residual_embeddings
        self.embed = False
        if n_vocab is not None:
            self.embed = True
            self.vectors = nn.Embedding(n_vocab, 300)
            if vectors is not None:
                self.vectors.weight.data = vectors
        self.rnn0 = nn.LSTM(300, 300, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn1 = nn.LSTM(600, 300, num_layers=1, bidirectional=True, batch_first=True)

        state_dict = model_zoo.load_url(WMT_LSTM, model_dir=MODEL_CACHE)
        layer0_dict = {k: v for k, v in state_dict.items() if 'l0' in k}
        layer1_dict = {k.replace('l1', 'l0'): v for k, v in state_dict.items() if 'l1' in k}
        self.rnn0.load_state_dict(layer0_dict)
        self.rnn1.load_state_dict(layer1_dict)

    def forward(self, inputs, lengths, hidden=None):
        """
        Arguments:
            inputs (Tensor): If MTLSTM handles embedding, a Long Tensor of size (batch_size, timesteps).
                             Otherwise, a Float Tensor of size (batch_size, timesteps, features).
            lengths (Long Tensor): lenghts of each sequence for handling padding
            hidden (Float Tensor): initial hidden state of the LSTM
        """
        if self.embed:
            inputs = self.vectors(inputs)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths).long()
            if inputs.is_cuda:
                with torch.cuda.current_device():
                    lengths = lengths.cuda()
        lens, indices = torch.sort(lengths, 0, True)
        outputs = [inputs] if self.residual_embeddings else []
        len_list = lens.tolist()
        packed_inputs = pack(inputs[indices], len_list, batch_first=True)

        outputs0, hidden_t0 = self.rnn0(packed_inputs, hidden)
        unpacked_outputs0 = unpack(outputs0, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        unpacked_outputs0 = unpacked_outputs0[_indices]
        # outputs.append(unpacked_outputs0)
        packed_inputs = outputs0

        outputs1, hidden_t1 = self.rnn1(packed_inputs, hidden)
        unpacked_outputs1 = unpack(outputs1, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        unpacked_outputs1 = unpacked_outputs1[_indices]
        outputs.append(unpacked_outputs1)

        outputs = torch.cat(outputs, 2) # Concatenation on 2nd dim
        return outputs.detach()
