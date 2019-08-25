import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class MTENCODER(nn.Module):
    

