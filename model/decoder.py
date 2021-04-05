import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionGraphConv(nn.Module):
    def __init__(self, supports_len, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, filter_type, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = supports_len*max_diffusion_step + 1  # add itself, i.e. x0
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self.output_dim = output_dim
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
        self.filter_type = filter_type

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, supports, inputs, state, bias_start=0.0):
        """
        Diffusion Graph convolution
        :param inputs: tensor, [batch_size, num_nodes, input_dim]
        :param state: tensor, [B, num_nodes, num_units]
        :param bias_start:
        :return: tensor, [batch_size, num_nodes, output_size]
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)
        x0_ori = x0

        if self._max_diffusion_step == 0:
            pass
        else:
            # if self.filter_type == "laplacian":
            #     for support in supports:
            #         x1 = torch.sparse.mm(support, x0)
            #         x = self._concat(x, x1)
            #         for k in range(2, self._max_diffusion_step + 1):
            #             x2 = 2 * torch.sparse.mm(support, x1) - x0
            #             x = self._concat(x, x2)
            #             x1, x0 = x2, x1

            # elif self.filter_type == "dual_random_walk" or self.filter_type == "identity":
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = torch.sparse.mm(support, x1)
                    x = self._concat(x, x2)
                    x1 = x2

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes, self.output_dim])


class MultiplicativeAttention(nn.Module):
    def __init__(self, num_nodes, enc_hid_dim, dec_hid_dim, attn_dim):
        """
        :param num_nodes: used for deciding the value of scale
        :param enc_hid_dim:
        :param dec_hid_dim: should be the same as enc_hid_dim
        :param attn_dim: linear transform the input to this dimension
        """
        super().__init__()
        assert enc_hid_dim == dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        # Perform linear transform.
        self.W_q = nn.Linear(dec_hid_dim, attn_dim)
        self.W_k = nn.Linear(enc_hid_dim, attn_dim)
        # self.W_v = nn.Linear(enc_hid_dim, enc_hid_dim)
        # self.scale = torch.sqrt(torch.FloatTensor([num_nodes*attn_dim]))
        self.scale = math.sqrt(num_nodes*attn_dim)

    def forward(self, dec_hidden, encoder_outputs):
        """
        :param dec_hidden: [batch_size, num_nodes, dec_hid_dim]
        :param encoder_outputs: should be [batch_size, time_steps, num_nodes, c_out].
        but somehow it is actually [batch_size*time_step, num_nodes*enc_hid_dim]
        :return: attention vector, [batch size, src len]
        """
        q = dec_hidden
        k = v = encoder_outputs
        batch_size, time_step, n_route, _ = encoder_outputs.shape

        # Compute attention for each single node
        q = torch.reshape(q, [batch_size * n_route, -1]).unsqueeze(1)  # [64*207, 1, 4]
        # k = torch.transpose(k, 1, 2)
        # k = torch.transpose(k, 2, 3)
        k = k.permute([0, 2, 3, 1])
        k = torch.reshape(k, [batch_size * n_route, -1, time_step])  # [64*207, 4, 12]
        v = torch.transpose(v, 1, 2).reshape([batch_size * n_route, time_step, -1])  # (64*207, 12, 4)
        # print("q shape: ", q.shape)
        # print("k shape: ", k.shape)
        # print("v shape: ", v.shape)

        energy = torch.matmul(q, k) / self.scale
        # 单个node的情况： (64*207, 1, 12)

        attention = F.softmax(energy, dim=-1)  # (64*207, 1, 12)
        weighted = torch.bmm(attention, v)
        # [batch_size*n_route, trg_len=1, src_len] * [batch_size*n_route, src_len, hidden_size]
        # = (64*207, 1, 4)

        return attention, weighted


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, input_dim, num_units, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, filter_type='dual_random_walk'):
        """
        :param num_units: the hidden dim of rnn
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._supports = []
        # if filter_type == "laplacian":
        #     supports_len = 1
        # elif filter_type == "random_walk":
        #     supports_len = 1
        if filter_type == "dual_random_walk":
            supports_len = 2
        elif filter_type == "identity":
            supports_len = 1
        else:
            raise ValueError("Unknown filter type...")

        self.dconv_gate = DiffusionGraphConv(supports_len=supports_len, input_dim=input_dim,
                                             hid_dim=num_units, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2, filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(supports_len=supports_len, input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units, filter_type=filter_type)
        if num_proj is not None:
            self.project1 = nn.Linear(self._num_units, self._num_units)
            self.project2 = nn.Linear(self._num_units, self._num_proj)
            # self.dropout = nn.Dropout(0.5)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, supports, inputs, state):
        """
        :param supports
        :param inputs: (B, num_nodes, input_dim)
        :param state: (B, num_nodes, num_units)
        :return: [B, num_nodes * num_units]  or output: [B, num_nodes * output_dim](with projection)
        """
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        value = torch.sigmoid(self.dconv_gate(supports, inputs, state, bias_start=1.0))  # (50, 228, 64)
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        c = self.dconv_candidate(supports, inputs, r * state)  # batch_size, self._num_nodes, output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            output = torch.reshape(self.project2(torch.relu(self.project1(output))),
                                   shape=(batch_size, self.output_size))  # (64, 207*1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


class Decoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 enc_hid_dim, dec_hid_dim, output_dim, attention_type, filter_type):
        super().__init__()
        self.dec_hid_dim = dec_hid_dim
        self.enc_hid_dim = enc_hid_dim
        self._num_nodes = num_nodes  # 207
        self._output_dim = output_dim  # should be 1

        self.decoding_cell = DCGRUCell(input_dim=input_dim+dec_hid_dim, num_units=dec_hid_dim,
                                       max_diffusion_step=max_diffusion_step,
                                       num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

        if attention_type == "multiplicative":
            self.attention = MultiplicativeAttention(num_nodes=num_nodes, enc_hid_dim=enc_hid_dim,
                                                     dec_hid_dim=dec_hid_dim, attn_dim=dec_hid_dim)
            self.decoding_cell = DCGRUCell(input_dim=input_dim + dec_hid_dim, num_units=dec_hid_dim,
                                           max_diffusion_step=max_diffusion_step,
                                           num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)
        elif attention_type == "no_attention":
            self.attention = None
            self.decoding_cell = DCGRUCell(input_dim=input_dim, num_units=dec_hid_dim,
                                           max_diffusion_step=max_diffusion_step,
                                           num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)
        else:
            raise ValueError("Unknown attention type")

    def forward(self, supports, inputs, initial_hidden_state, encoder_outputs, teacher_forcing_ratio=0.5):
        """
        :param supports
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the output context of the encoder. [batch_size, n_route, dec_hid_dim]
        :param encoder_outputs: [batch_size, time_steps, n_route, c_out]
        :param teacher_forcing_ratio:
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
        """
        _, enc_seq_length, _, _ = encoder_outputs.shape
        seq_length, batch_size, num_nodes, _ = inputs.shape

        # tensor to store decoder outputs
        outputs = torch.zeros(seq_length, batch_size, self._num_nodes*self._output_dim)  # (13, 50, 207*1)

        current_input = inputs[0]  # the first input to the rnn is Start Symbol
        hidden_state = initial_hidden_state  # [batch_size, num_nodes, dec_hid_dim]
        temporal_attn = torch.FloatTensor([]).cuda()

        for t in range(1, seq_length):
            current_input = torch.reshape(current_input, [batch_size, num_nodes, -1])
            if self.attention is not None:
                a, weighted = self.attention(hidden_state, encoder_outputs)  # [batch_size, 1, src len]
                a = torch.reshape(a, [batch_size, num_nodes, 1, -1])
                weighted = torch.reshape(weighted.squeeze(), [batch_size, num_nodes, -1])
                rnn_input = torch.cat((current_input, weighted), dim=2)  # (batch_size, num_nodes, in_dim+dec_dim)
                temporal_attn = torch.cat([temporal_attn, a], dim=2)
            else:
                rnn_input = current_input
            output, hidden_state = self.decoding_cell(supports, rnn_input, hidden_state)
            outputs[t] = output  # (batch_size, num_nodes*out_dim)
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)
        return outputs[1:, ...], temporal_attn


# GRUDecoder
class GRUDecoder(nn.Module):
    """
    A GRU decoder. Add tricks like scheduled sampling.
    Alleviate information compression.
    ignore spatial factors.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.decoding_cell = nn.GRUCell(input_size=input_dim+hidden_dim, hidden_size=hidden_dim, bias=True)
        self.proj1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.proj2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.attention = MultiplicativeAttention(num_nodes=num_nodes, enc_hid_dim=hidden_dim,
                                                 dec_hid_dim=hidden_dim, attn_dim=hidden_dim)

    def forward(self, inputs, initial_hidden_state, encoder_outputs, teacher_forcing_ratio=0.5):
        """
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the output context of the encoder. [batch_size, num_nodes, dec_hid_dim]
        :param encoder_outputs: [batch_size, time_steps, num_nodes, dec_hid_dim]
        :param teacher_forcing_ratio: changes through time.
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 64, 207*1)
        """
        _, enc_seq_length, _, _ = encoder_outputs.shape
        seq_length, batch_size, num_nodes, _ = inputs.shape

        # tensor to store decoder outputs
        outputs = torch.zeros(seq_length, batch_size, num_nodes * self.output_dim)  # (13, 50, 207*1)

        # Alleviate information compression
        initial_hidden_state = torch.reshape(initial_hidden_state, (batch_size * num_nodes, -1))

        current_input = inputs[0]  # the first input to the rnn is GO Symbol
        hidden_state = initial_hidden_state  # [batch_size, num_nodes, hidden_dim]
        for t in range(1, seq_length):
            # GRUCell input should be (batch_size, input_size), hidden should be (batch_size, hidden_size)
            current_input = torch.reshape(current_input, (batch_size * num_nodes, -1))
            a, weighted = self.attention(hidden_state.reshape((batch_size, num_nodes, -1)), encoder_outputs)
            # [batch_size, 1, src len]
            weighted = torch.reshape(weighted.squeeze(1), [batch_size*num_nodes, -1])
            rnn_input = torch.cat((current_input, weighted), dim=-1)
            hidden_state = self.decoding_cell(rnn_input, hidden_state)  # [batch_size*num_nodes, dec_hid_dim]
            output = self.proj2(torch.relu(self.proj1(hidden_state)))  # [batch_size*num_nodes, output_dim]
            outputs[t] = torch.reshape(output.squeeze(), (batch_size, num_nodes*self.output_dim))
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)
        return outputs[1:, ...]
