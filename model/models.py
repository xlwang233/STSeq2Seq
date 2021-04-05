import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder, GRUDecoder


class STSeq2Seq(nn.Module):
    def __init__(self, batch_size, kt, blocks, dec_input_dim, max_diffusion_step,
                 num_nodes, rnn_units, seq_len, output_dim, attention_type, filter_type, adpt_type):
        super().__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes  # should be 207
        self._rnn_units = rnn_units  # should be 64
        self._seq_len = seq_len  # should be 12
        self.output_dim = output_dim  # should be 1
        self.dec_input_dim = dec_input_dim

        self.encoder = Encoder(batch_size=batch_size, num_nodes=num_nodes,
                               his_len=seq_len, max_diffusion_step=max_diffusion_step,
                               kt=kt, blocks=blocks, dec_hid_dim=rnn_units,
                               filter_type=filter_type, adpt_type=adpt_type)
        self.decoder = Decoder(input_dim=dec_input_dim,
                               max_diffusion_step=max_diffusion_step,
                               num_nodes=num_nodes, enc_hid_dim=blocks[-1][-1],
                               dec_hid_dim=rnn_units, output_dim=output_dim,
                               attention_type=attention_type,
                               filter_type=filter_type)

    def forward(self, supports, source, target, teacher_forcing_ratio):
        # the size of source/target would be (64, 12, 207, 2)
        target = torch.transpose(target[..., :self.output_dim], dim0=0, dim1=1)
        Start = torch.zeros(1, self.batch_size, self.num_nodes, self.dec_input_dim, device=source.device)
        target = torch.cat([Start, target], dim=0)

        encoder_outputs, hidden = self.encoder(supports, source)

        outputs, temporal_attn = self.decoder(supports=supports, inputs=target, initial_hidden_state=hidden,
                               encoder_outputs=encoder_outputs, teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs, temporal_attn  # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)


# GRU
class GRUSeq2Seq(nn.Module):
    def __init__(self, batch_size, num_nodes, hidden_size, enc_input_dim, dec_input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.dec_input_dim = dec_input_dim
        self.encoder = nn.GRU(input_size=enc_input_dim, hidden_size=hidden_size)

        self.decoder = GRUDecoder(input_dim=dec_input_dim, hidden_dim=hidden_size,
                                  output_dim=output_dim, num_nodes=num_nodes)

    def forward(self, supports, source, targets, teacher_forcing_ratio):
        """
        :param supports: None
        :param source: (batch_size, src_len, num_nodes, input_dim)
        :param targets: (batch_size, trg_len, num_nodes, input_dim)
        :param teacher_forcing_ratio
        :return: (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)
        """""
        batch_size, src_len, num_nodes, _ = source.shape
        source = torch.transpose(source, dim0=0, dim1=1)  # (12, 64, 207, 2)
        source = torch.reshape(source, [src_len, batch_size*num_nodes, -1])

        targets = torch.transpose(targets[..., :self.output_dim], dim0=0, dim1=1)  # (12, 64, 207, 1)
        Start = torch.zeros(1, self.batch_size, self.num_nodes, self.dec_input_dim, device=source.device)
        targets = torch.cat([Start, targets], dim=0)  # (13, )

        encoder_outputs, hidden = self.encoder(source)

        encoder_outputs = torch.reshape(encoder_outputs, [12, batch_size, num_nodes, -1])
        encoder_outputs = torch.transpose(encoder_outputs, dim0=0, dim1=1)
        hidden = torch.reshape(hidden, [batch_size, num_nodes, -1])

        outputs = self.decoder(inputs=targets, initial_hidden_state=hidden,
                               encoder_outputs=encoder_outputs,
                               teacher_forcing_ratio=teacher_forcing_ratio)

        return outputs  # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)


# FNN
class Hidden(nn.Module):
    def __init__(self, num_nodes, in_features, out_features):
        super(Hidden, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(size=(num_nodes, in_features, out_features)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(num_nodes, out_features)))
        nn.init.xavier_normal_(self.weights.data, gain=1.414)
        nn.init.constant_(self.biases, val=0.0)

    def forward(self, x):
        """
        :param x: input size: (batch_size, num_nodes, in_features)
        :return: (batch_size, num_nodes, out_features)
        """
        # (batch_size, num_nodes, in_features) * (num_nodes, in_features, out_features)
        y = torch.einsum('abc,bcd->abd', x, self.weights)  # (batch_size, num_nodes, out_features)
        y = torch.add(y, self.biases)  # broadcastable  (b, m, n) + (m, n) = (b, m, n)
        return y


class FNN(nn.Module):
    """
    Feed forward neural networks with two hidden layers, each layer contains 256 units.
    """
    def __init__(self, batch_size, num_nodes, seq_len, input_dim, hidden_size, horizon, output_dim):
        super(FNN, self).__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.fc1 = Hidden(num_nodes, seq_len*input_dim, hidden_size)
        self.fc2 = Hidden(num_nodes, hidden_size, hidden_size)
        self.proj = nn.Linear(in_features=hidden_size, out_features=horizon)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        :param x: (batch_size, his_len, num_nodes, input_dim)
        :return:
        """
        batch_size, his_len, num_nodes, input_dim = x.shape
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (batch_size, num_nodes, -1))
        y = torch.relu(self.dropout(self.fc1(x)))
        y = torch.relu(self.dropout(self.fc2(y)))  # (batch_size, num_nodes, hidden_size)
        y = torch.reshape(y, (batch_size*num_nodes, -1))
        y = self.proj(y)  # (batch_size*num_nodes, horizon)
        y = torch.reshape(y, (batch_size, num_nodes, -1))
        y = torch.transpose(y, 1, 2)
        return y  # (batch_size, horizon, num_nodes)  (64, 12, 207)
