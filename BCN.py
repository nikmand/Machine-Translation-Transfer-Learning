import torch
import torch.nn as nn
from MaxOut import Maxout
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from Constants import mtlstm_hidden_size

from encoder import MTLSTM


class BCN(nn.Module):
    """implementation of Biattentive Classification Network in
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""

    # ToDo :
    #  include 2 sentences case,
    #  review masks + parameters

    def __init__(self, config, n_vocab, vocabulary, embeddings, num_labels, embeddings_type):
        super(BCN, self).__init__()
        self.word_vec_size = config['word_vec_size']
        self.mtlstm_hidden_size =  mtlstm_hidden_size[embeddings_type] # config['mtlstm_hidden_size']
        self.cove_size = self.word_vec_size + self.mtlstm_hidden_size
        self.fc_hidden_size = config['fc_hidden_size']
        self.bilstm_encoder_size = config['bilstm_encoder_size']
        self.bilstm_integrator_size = config['bilstm_integrator_size']
        self.embeddings_type = embeddings_type
        # self.decove = config["deep_vectors"]

        self.mtlstm = MTLSTM(n_vocab=n_vocab, vectors=vocabulary, residual_embeddings=True, model_cache=embeddings,
                             layer0=True, layer1=True)

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

        self.pool_size = config["maxout_channels"]
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(config['dropout'])
        # self.fc1 = nn.Linear(self.bilstm_integrator_size * 4,
        #                      self.bilstm_integrator_size * 4 // 4)
        # self.fc2 = nn.Linear(self.bilstm_integrator_size * 4 // 4, self.bilstm_integrator_size * 4 // 4 // 4)
        # self.classifier = nn.Linear(self.bilstm_integrator_size * 4 // 4 // 4, num_labels)

        self.bn1 = nn.BatchNorm1d(self.bilstm_integrator_size * 4)
        self.bn2 = nn.BatchNorm1d((self.bilstm_integrator_size * 4) // 4)
        self.maxout1 = Maxout(self.bilstm_integrator_size * 4, self.bilstm_integrator_size * 4 // 4, self.pool_size)
        self.maxout2 = Maxout(self.bilstm_integrator_size * 4 // 4,
                              self.bilstm_integrator_size * 4 // 4 // 4, self.pool_size)
        self.classifier = nn.Linear((self.bilstm_integrator_size * 4) // 4 // 4, num_labels)

        self.device = config['device']

        self.w = nn.Parameter(torch.Tensor([[0.0], [0.0]]), requires_grad=True)
        self.gama = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def makeMask(self, lens, hidden_size):
        mask = []
        lens = lens.tolist()
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
        if self.device != 'cpu':
            return trans_mask.cuda()
        else:
            return trans_mask

    def forward(self, tokens_emb, length):

        reps = self.mtlstm(tokens_emb, length) # size 1500

        glove = reps[:, :, :300]
        cove_1 = reps[:, :, 300:900]
        cove_2 = reps[:, :, 900:]

        if self.embeddings_type == 'glove':
            reps = glove
        elif self.embeddings_type == 'cove_1':
            reps = torch.cat([glove, cove_1], dim=2)
        elif self.embeddings_type == 'cove_2':
            reps = torch.cat([glove, cove_2], dim=2)
        elif self.embeddings_type == 'decove':
            s = torch.cat([cove_1.unsqueeze(-1), cove_2.unsqueeze(-1)], 3).detach()
            softmax_weghts = torch.softmax(self.w, dim=0)
            weighted_reps = s.matmul(softmax_weghts).squeeze(-1) * self.gama
            reps = torch.cat([glove, weighted_reps], dim=2) # size 300 + 600

        print("REPS {}".format(reps.shape))
        reps = self.dropout(reps)

        task_specific_reps = (self.relu(self.fc(reps)))

        lens, indices = torch.sort(length, 0, True)
        len_list = lens.tolist()
        task_specific_reps = pack(task_specific_reps[indices], len_list, batch_first=True)

        outputs, _ = self.bilstm_encoder(task_specific_reps)
        X, _ = unpack(outputs, batch_first=True)
        _, _indices = torch.sort(indices, 0)
        X = X[_indices]

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = X.bmm(X.permute(0, 2, 1))

        # turn small values into very negative ones (-Inf) so that they can be zeroed during softmax
        attention_mask1 = torch.tensor((-1e32 * (attention_logits <= 1e-7).float()).detach(),
                                       device=torch.device(self.device))
        masked_attention_logits = attention_logits + attention_mask1  # mask logits that are near zero
        masked_Ax = self.sm(masked_attention_logits)  # prerform column-wise softmax
        masked_Ay = self.sm(masked_attention_logits.permute(0, 2, 1))
        attention_mask2 = torch.tensor((attention_logits >= 1e-7).float().detach(),
                                       device=torch.device(self.device))  # mask those all zeros
        Ax = masked_Ax * attention_mask2
        Ay = masked_Ay * attention_mask2

        Cx = torch.bmm(Ax.permute(0, 2, 1), X)  # batch_size * max_len * bilstm_encoder_size
        Cy = torch.bmm(Ay.permute(0, 2, 1), X)

        # Build the input to the integrator
        integrator_input = torch.cat([X,
                                      X - Cy,
                                      X * Cy], 2)

        integrator_input = pack(integrator_input[indices], len_list, batch_first=True)

        outputs, _ = self.bilstm_integrator(integrator_input)  # batch_size * max_len * bilstm_integrator_size
        Xy, _ = unpack(outputs, batch_first=True)
        _, _indices = torch.sort(indices, 0)
        Xy = Xy[_indices]

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
        self_attentive_logits = self.attentive_pooling_proj(Xy)
        self_attentive_logits = torch.squeeze(self_attentive_logits) \
                                + -1e32 * (1 - self.makeMask(length, 1))
        self_weights = self.sm(self_attentive_logits)
        self_attentive_pool = self_weights.unsqueeze(1).bmm(Xy).squeeze(1)

        pooled_representations = torch.cat([max_pool,
                                            min_pool,
                                            mean_pool,
                                            self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout(pooled_representations)

        bn_pooled = self.bn1(pooled_representations_dropped)
        max_out1 = self.maxout1(bn_pooled)
        max_out1_dropped = self.dropout(max_out1)
        bn_max_out1 = self.bn2(max_out1_dropped)
        max_out2 = self.maxout2(bn_max_out1)
        max_out2_dropped = self.dropout(max_out2)

        # rep = self.dropout(self.relu(self.fc1(pooled_representations_dropped)))
        # rep = self.dropout(self.relu(self.fc2(rep)))

        logits = self.classifier(max_out2_dropped)
        return logits
