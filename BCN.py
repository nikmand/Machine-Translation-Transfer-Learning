import torch
import torch.nn as nn
from MaxOut import Maxout
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from encoder import MTLSTM


class BCN(nn.Module):
    """implementation of Biattentive Classification Network in
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""

    # ToDo :
    #  include 2 sentences case,
    #  review masks + maxout + parameters

    def __init__(self, config, n_vocab, vocabulary, embeddings, num_labels):
        super(BCN, self).__init__()
        self.word_vec_size = config['word_vec_size']
        self.mtlstm_hidden_size = config['mtlstm_hidden_size']
        self.cove_size = self.mtlstm_hidden_size + self.word_vec_size
        self.fc_hidden_size = config['fc_hidden_size']
        self.bilstm_encoder_size = config['bilstm_encoder_size']
        self.bilstm_integrator_size = config['bilstm_integrator_size']

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

        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(config['dropout'])
        self.bn1 = nn.BatchNorm1d(self.bilstm_integrator_size * 4)
        self.bn2 = nn.BatchNorm1d((self.bilstm_integrator_size * 4)//2//2)
        self.fc1_maxout = nn.Linear(self.bilstm_integrator_size * 4, (self.bilstm_integrator_size * 4)//2)
        self.fc2_maxout = nn.Linear((self.bilstm_integrator_size * 4) //2 //2,
                                    (self.bilstm_integrator_size * 4) // 2 // 2 // 2)
        self.classifier = nn.Linear((self.bilstm_integrator_size * 4) // 2 // 2 // 2 // 2, num_labels)
        self.maxout = Maxout()

        self.gpu = config['gpu']

    def makeMask(self, lens, hidden_size):
        mask = []
        max_len = max(lens)
        for l in lens:
            mask.append([1] * l + [0] * (max_len - l))
        mask = torch.FloatTensor(mask)
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

        reps = self.mtlstm(tokens_emb, length)
        reps = self.dropout(reps)

        task_specific_reps = (self.relu(self.fc(reps)))

        lens, indices = torch.sort(length, 0, True)
        len_list = lens.tolist()
        task_specific_reps = pack(task_specific_reps[indices], len_list, batch_first=True)

        outputs, _ = self.bilstm_encoder(task_specific_reps)
        X, _ = unpack(outputs, batch_first=True)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = X.bmm(X.permute(0, 2, 1))

        attention_mask1 = torch.Tensor((-1e7 * (attention_logits <= 1e-7).float()).detach())
        masked_attention_logits = attention_logits + attention_mask1  # mask logits that are near zero
        masked_Ax = self.sm(masked_attention_logits)  # prerform column-wise softmax
        masked_Ay = self.sm(masked_attention_logits.permute(0, 2, 1))
        attention_mask2 = torch.Tensor((attention_logits >= 1e-7).float().detach())  # mask those all zeros
        Ax = masked_Ax * attention_mask2
        Ay = masked_Ay * attention_mask2

        Cx = torch.bmm(Ax.permute(0, 2, 1), X)  # batch_size * max_len * bilstm_encoder_size
        Cy = torch.bmm(Ay.permute(0, 2, 1), X)

        # Build the input to the integrator
        integrator_input = torch.cat([Cy,
                                      X - Cy,
                                      X * Cy], 2)

        integrator_input = pack(integrator_input[indices], len_list, batch_first=True)

        outputs, _ = self.bilstm_integrator(integrator_input)  # batch_size * max_len * bilstm_integrator_size
        Xy, _ = unpack(outputs, batch_first=True)

        # Simple Pooling layers
        max_masked_Xy = Xy + -1e7 * (1 - self.makeMask(len_list,
                                                       self.bilstm_integrator_size))
        max_pool = torch.max(max_masked_Xy, 1)[0]
        min_masked_Xy = Xy + 1e7 * (1 - self.makeMask(len_list,
                                                      self.bilstm_integrator_size))
        min_pool = torch.min(min_masked_Xy, 1)[0]
        mean_pool = torch.sum(Xy, 1) / torch.sum(self.makeMask(len_list, 1),
                                                 1,
                                                 keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attentive_pooling_proj(Xy)
        self_attentive_logits = torch.squeeze(self_attentive_logits) \
                                + -1e7 * (1 - self.makeMask(len_list, 1))
        self_weights = self.sm(self_attentive_logits)
        self_attentive_pool = torch.bmm(self_weights.unsqueeze(1),
                                        Xy).squeeze(1)

        pooled_representations = torch.cat([max_pool,
                                            min_pool,
                                            mean_pool,
                                            self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout(pooled_representations)

        bn_pooled = self.bn1(pooled_representations_dropped)
        max_out1 = self.maxout(self.fc1_maxout(bn_pooled))
        max_out1_dropped = self.dropout(max_out1)
        bn_max_out1 = self.bn2(max_out1_dropped)
        max_out2 = self.maxout(self.fc2_maxout(bn_max_out1))
        max_out2_dropped = self.dropout(max_out2)

        logits = self.classifier(max_out2_dropped)
        return logits
