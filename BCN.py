import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from encoder import MTLSTM


class BCN(nn.Module):
    """implementation of Biattentive Classification Network in
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""

    def __init__(self, config, n_vocab, vocabulary, embeddings):
        super(BCN, self).__init__()
        self.word_vec_size = config['word_vec_size']
        self.mtlstm_hidden_size = config['mtlstm_hidden_size']
        self.cove_size = self.mtlstm_hidden_size + self.word_vec_size
        self.fc_hidden_size = config['fc_hidden_size']
        self.bilstm_encoder_size = config['bilstm_encoder_size']
        self.bilstm_integrator_size = config['bilstm_integrator_size']
        self.fc_hidden_size1 = config['fc_hidden_size1']
        self.mem_size = config['mem_size']

        self.mtlstm = MTLSTM(n_vocab=n_vocab, vectors=vocabulary, residual_embeddings=True, model_cache=embeddings)

        self.fc = nn.Linear(self.cove_size, self.fc_hidden_size)

        self.bilstm_encoder = nn.LSTM(self.fc_hidden_size,
                                      self.bilstm_encoder_size // 2,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=config['dropout'])

        self.bilstm_integrator = nn.LSTM(self.bilstm_encoder_size * 3,
                                         self.bilstm_integrator_size // 2,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=True,
                                         dropout=config['dropout'])

        self.attentive_pooling_proj = nn.Linear(self.bilstm_integrator_size,
                                                1)

        self.fc1 = nn.Linear(self.bilstm_integrator_size * 4,
                             self.fc_hidden_size1)
        self.fc2 = nn.Linear(self.fc_hidden_size1, self.mem_size)

        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        self.log_sm = nn.LogSoftmax()
        self.dropout = nn.Dropout(config['dropout'])

        self.gpu = config['gpu']

    def makeMask(self, lens, hidden_size):
        mask = []
        max_len = max(lens)
        for l in lens:
            mask.append([1] * l + [0] * (max_len - l))
        mask = Variable(torch.FloatTensor(mask))
        if hidden_size == 1:
            trans_mask = mask
        else:
            trans_mask = mask.unsqueeze(2).expand(mask.size(0),
                                                  mask.size(1),
                                                  hidden_size)
        if self.gpu != -1:
            return trans_mask.cuda()
        else:
            return trans_mask

    def forward(self, tokens_emb, length):
        batch_size = tokens_emb.size(0)

        reps = self.mtlstm(tokens_emb, length)
        reps = self.dropout(reps)

        max_len = max(length)

        compressed_reps = reps.view(-1, self.cove_size)
        task_specific_reps = (self.relu(self.fc(compressed_reps))).view(
            batch_size,
            max_len,
            self.fc_hidden_size)
        task_specific_reps = pack(task_specific_reps,
                                  length,
                                  batch_first=True)

        outputs, _ = self.bilstm_encoder(task_specific_reps)
        X, _ = unpack(outputs, batch_first=True)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = X.bmm(X.permute(0, 2, 1).contiguous())

        attention_mask1 = Variable((-1e7 * (attention_logits <= 1e-7).float()).data)
        masked_attention_logits = attention_logits + attention_mask1
        compressed_Ay = self.sm(masked_attention_logits.view(-1, max_len))
        attention_mask2 = Variable((attention_logits >= 1e-7).float().data)  # mask those all zeros
        Ay = compressed_Ay.view(batch_size, max_len, max_len) * attention_mask2

        Cy = torch.bmm(Ay, X)  # batch_size * max_len * bilstm_encoder_size

        # Build the input to the integrator
        integrator_input = torch.cat([Cy,
                                      X - Cy,
                                      X * Cy], 2)
        integrator_input = pack(integrator_input, length, batch_first=True)

        outputs, _ = self.bilstm_integrator(integrator_input)  # batch_size * max_len * bilstm_integrator_size
        Xy, _ = unpack(outputs, batch_first=True)

        # Simple Pooling layers
        max_masked_Xy = Xy + -1e7 * (1 - self.makeMask(length,
                                                       self.bilstm_integrator_size))
        max_pool = torch.max(max_masked_Xy, 1)[0]
        min_masked_Xy = Xy + 1e7 * (1 - self.makeMask(length,
                                                      self.bilstm_integrator_size))
        min_pool = torch.min(min_masked_Xy, 1)[0]
        mean_pool = torch.sum(Xy, 1) / torch.sum(self.makeMask(length, 1),
                                                 1,
                                                 keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attentive_pooling_proj(Xy.contiguous().view(-1,
                                                                                 self.bilstm_integrator_size))
        self_attentive_logits = self_attentive_logits.view(batch_size, max_len) \
                                + -1e7 * (1 - self.makeMask(length, 1))
        self_weights = self.sm(self_attentive_logits)
        self_attentive_pool = torch.bmm(self_weights.view(batch_size,
                                                          1,
                                                          max_len),
                                        Xy).squeeze(1)

        pooled_representations = torch.cat([max_pool,
                                            min_pool,
                                            mean_pool,
                                            self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout(pooled_representations)

        rep = self.dropout(self.relu(self.fc1(pooled_representations_dropped)))
        rep = self.dropout(self.relu(self.fc2(rep)))

        return rep, None
