import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncGraphConv(nn.Module):
    def __init__(self, supports_len, batch_size, num_nodes, his_len, d_in,
                 max_diffusion_step, d_out, filter_type, bias_start=0.0, adpt_type="pam"):
        """
        graph convolution for ST-conv encoder
        :param supports_len:
        :param batch_size: here batch_size = batch_size * time_steps
        :param num_nodes: number of nodes in graph
        :param d_in: int, input dimension.
        :param d_out: output_dim, the size of output channel.
        :param bias_start:
        """
        super().__init__()
        if adpt_type == "no_adpt":
            self.num_matrices = supports_len * max_diffusion_step + 1
        else:  # "pam" or "random_embedding"
            self.num_matrices = supports_len*max_diffusion_step + 2
        self.batch_size = batch_size
        self.his_len = his_len
        self.batch_size_t = batch_size * his_len
        self.d_in = d_in
        self.d_out = d_out
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(size=(d_in*self.num_matrices, d_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(d_out,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
        self.filter_type = filter_type
        self.adpt_type = adpt_type

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, supports, x, adpt_adj=None):
        """
        :param supports:
        :param x: tensor, [batch_size * time_step * d_in, num_nodes]. treat batch_size*time_step as batch_size
        :param x: tensor, [batch_size, time_step, n_route, d_in]
        :return: tensor, [batch_size, num_nodes, d_out].
        """
        # gcn with PAM
        if self.adpt_type == "pam":
            x_add = torch.reshape(x, [self.batch_size, self.his_len, self.d_in, self._num_nodes]).permute(0, 3, 1, 2)
            x_add = torch.reshape(x_add, [self.batch_size, self._num_nodes, -1])
            x_add = torch.reshape(torch.bmm(adpt_adj, x_add), [self.batch_size, self._num_nodes, self.his_len, -1])
            # (64, 207, 207) * (64, 207, 12*32) = (64, 207, 12*32) and then reshape to ()
            x_add = torch.reshape(torch.transpose(x_add, 0, 1),
                              [self._num_nodes, self.batch_size * self.his_len*self.d_in])  # (207,50*12*32)
        elif self.adpt_type == "random_embedding":
            # adpt_adj (num_nodes, num_nodes)
            x_add = torch.transpose(x, dim0=0, dim1=1)  # [num_nodes, batch_size * time_step * d_in]
            x_add = torch.matmul(adpt_adj, x_add)
        else:
            x_add = None

        # Diffusion convolution
        x0 = torch.transpose(x, dim0=0, dim1=1)  # [num_nodes, batch_size * time_step * d_in]
        x = torch.unsqueeze(x0, dim=0)
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = torch.sparse.mm(support, x1)
                    x = self._concat(x, x2)
                    x1 = x2
        if x_add is not None:
            x = self._concat(x, x_add)
        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, self.batch_size_t, self.d_in])
        x = torch.transpose(x, dim0=0, dim1=2)  # (batch_size, num_nodes, order, input_dim)
        x = torch.reshape(x, shape=[self.batch_size_t*self._num_nodes, self.num_matrices*self.d_in])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, d_out)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [self.batch_size_t, self._num_nodes, self.d_out])


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, d_in, d_out):
        """
        Temporal convolution layer.
        :param kt: int, kernel size of temporal convolution.
        :param d_in: int, size of input channel.
        :param d_out: int, size of output channel.
        """
        super().__init__()
        self.kt = kt
        self.d_in = d_in
        self.d_out = d_out
        if d_in > d_out:
            self.input_conv = nn.Conv2d(d_in, d_out, (1, 1))
        self.conv1 = nn.Conv2d(d_in, 2*d_out, (kt, 1), padding=(int((kt-1)/2), 0))
        # kernel size: [kernel_height, kernel_width]
        self.scale = math.sqrt(0.5)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, P, num_nodes, D_in].
        :return: tensor, [batch_size, P, num_nodes, D_out].
        Note the time_step is unchanged because we added padding.
        """

        # convolution input format: TensorFlow NHWC, PyTorch NCHW.
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, d_in, P, num_nodes]
        _, _, P, n = x.shape
        if self.d_in > self.d_out:
            x_input = self.input_conv(x)
        elif self.d_in < self.d_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.d_out-self.d_in, P, n], device=x.device)], dim=1)
        else:
            x_input = x

        # gated liner unit
        x_conv = self.conv1(x)
        out = (F.glu(x_conv, dim=1) + x_input) * self.scale
        out = out.permute(0, 2, 3, 1)
        return out


class SpatialConvLayer(nn.Module):
    def __init__(self, batch_size, num_nodes, d_in, d_out, max_diffusion_step,
                 filter_type, adpt_type="pam"):
        """
        Spatial  convolution layer.
        :param batch_size: int, should be batch_size * time_step
        :param num_nodes:
        :param d_in: int, size of input channel.
        :param d_out: int, size of output channel.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        if d_in > d_out:
            self.input_conv = nn.Conv2d(d_in, d_out, (1, 1))
        if filter_type == "dual_random_walk":
            supports_len = 2
        elif filter_type == "identity":
            supports_len = 1
        else:
            raise ValueError("unknown filter type")
        self.spatial_conv = EncGraphConv(supports_len=supports_len, batch_size=batch_size, num_nodes=num_nodes,
                                         his_len=12, d_in=d_in, max_diffusion_step=max_diffusion_step,
                                         d_out=d_out, filter_type=filter_type, bias_start=0.0, adpt_type=adpt_type)

    def forward(self, supports, x, adpt_adj=None):
        """
        :param supports: list of pre-computed weighted adj matrix.
        :param x: tensor, [batch_size, seq_len, num_nodes, d_in]. [batch_size, ]
        :return: tensor, [batch_size, seq_len, nun_nodes, d_out].
        """
        batch_size, seq_len, n, _ = x.shape

        # Convert tf NHWC into PyTorch NCHW format for pytorch to perform convolutions.
        x = x.permute(0, 3, 1, 2)  # [batch_size, d_in, seq_len, n_route]
        if self.d_in > self.d_out:
            x_input = self.input_conv(x)
        elif self.d_in < self.d_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.d_out - self.d_in, seq_len, n],
                                                device=x.device)], dim=1)
        else:
            x_input = x

        x = torch.transpose(x, dim0=1, dim1=2)  # [batch_size, seq_len, d_in, n_route]
        x_conv = self.spatial_conv(supports, torch.reshape(x, (-1, n)), adpt_adj)  # [batch_size, n_route, d_out]
        x_conv = torch.reshape(x_conv, [batch_size, seq_len, n, self.d_out])
        x_input = x_input.permute(0, 2, 3, 1)

        return torch.relu(x_conv[:, :, :, 0:self.d_out] + x_input)


class STConvBlock(nn.Module):
    def __init__(self, batch_size, num_nodes, kt, channels, max_diffusion_step, filter_type, adpt_type="adpt"):
        """
        Spatial-temporal convolutional block, which contains a temporal gated convolution layer
        and a spatial convolution layer.
        :param num_nodes: int, used for initialize LayerNorm
        :param kt: int, kernel size of temporal convolution.
        :param channels: list, channel configs of a single st_conv block.
        """
        super().__init__()
        self.kt = kt
        self.c_si, self.c_t, self.c_oo = channels
        self.temporal_conv_layer = TemporalConvLayer(kt, self.c_si, self.c_t)
        self.spatial_conv_layer = SpatialConvLayer(batch_size=batch_size,
                                                   num_nodes=num_nodes, d_in=self.c_t, d_out=self.c_oo,
                                                   max_diffusion_step=max_diffusion_step,
                                                   filter_type=filter_type, adpt_type=adpt_type)
        self.layer_norm = nn.LayerNorm([num_nodes, self.c_oo])

    def forward(self, supports, x, adpt_adj=None):
        """
        :param supports:
        :param adpt_adj: the adaptively computed adjacency matrix.
        :param x: x: tensor, [batch_size, time_step, num_nodes, d_in].
        :return: tensor, [batch_size, time_step, num_nodes, d_out].
        """
        x_s = self.temporal_conv_layer(x)
        x_t = self.spatial_conv_layer(supports, x_s, adpt_adj)
        x_ln = self.layer_norm(x_t)
        return x_ln


class ProjLayer(nn.Module):
    def __init__(self, time_step, channel, dec_hid_dim):
        """
        Projection layer after 2 ST-Conv blocks. Used for obtaining the initial hidden state.
        :param time_step:
        :param channel:
        :param dec_hid_dim: decoder's hidden dimension
        """
        super(ProjLayer, self).__init__()
        self.channel = channel
        self.dec_hid_dim = dec_hid_dim
        self.temporal_conv = nn.Conv2d(channel, dec_hid_dim, (time_step, 1))

    def forward(self, x):
        """
        :param x: tensor, [batch_size, seq_len, num_nodes, channel].
        :return: tensor, [batch_size, num_nodes, dec_hid_dim].   dec_hid_dim should be 64
        """
        batch_size, _, _, channel = x.shape
        assert channel == self.channel, "there is something wrong with the dimension of input"
        # maps multi-steps to one.
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)  # (batch_size, _, 1, 207)
        x = torch.transpose(x.squeeze(), 1, 2)
        return torch.relu(x)


class PAM(nn.Module):
    def __init__(self, his_len, his_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=his_len, out_features=his_dim)
        self.linear2 = nn.Linear(in_features=his_dim, out_features=his_dim)
        # self.scale = torch.sqrt(torch.FloatTensor([his_dim]))

    def forward(self, feature_data):
        """
        :param feature_data: tensor, [batch_size, time_step, n_route]. the time series of each nodes
        :return: learned pattern aware adjacency matrices.
        """
        feature_data = torch.transpose(feature_data, 1, 2)  # to (batch_size, n_route, time_step)
        feature_data = torch.relu(self.linear2(torch.relu(self.linear1(feature_data))))
        # to (batch_size, n_route, his_dim)
        # h = torch.matmul(torch.transpose(feature_data, 1, 2), self.W)  # (N, F)  (F, F') = (N, F')
        pams = F.softmax(torch.bmm(feature_data, torch.transpose(feature_data, 1, 2)), dim=-1)
        # pams: (batch_size, num_nodes, num_nodes)

        # added for scaled dot-product
        # adj = torch.bmm(feature_data, torch.transpose(feature_data, 1, 2))  # (64, 207, 207)
        # adj = adj / self.scale
        return pams


class Encoder(nn.Module):
    def __init__(self, batch_size, num_nodes, his_len, max_diffusion_step, kt, blocks, dec_hid_dim,
                 filter_type, adpt_type="pam"):
        """
        :param batch_size:
        :param num_nodes: int, the number of nodes in the graph. should be 207 for METR-LA
        :param his_len: int, length of historical sequence for training.
        :param kt: int, kernel size of temporal convolution.
        :param blocks: list, channel configs of st_conv blocks.
        :param max_diffusion_step: its function resembles ks
        """
        super().__init__()
        self.his_len = his_len
        self.kt = kt
        self.blocks = blocks
        self.adpt_type = adpt_type
        if adpt_type == "pam":
            self.getpam= PAM(his_len=his_len, his_dim=dec_hid_dim)
        elif adpt_type == "random_embedding":
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 64), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(64, num_nodes), requires_grad=True)
        else:
            raise ValueError("Wrong adpt_type...")
        st_conv_block_list = list()
        for channels in blocks:
            st_conv_block_list.append(STConvBlock(batch_size=batch_size,
                                                  num_nodes=num_nodes, kt=kt, channels=channels,
                                                  max_diffusion_step=max_diffusion_step,
                                                  filter_type=filter_type,
                                                  adpt_type=adpt_type))

        self.st_conv_blocks = nn.ModuleList(st_conv_block_list)
        self.output_layer = ProjLayer(time_step=self.his_len, channel=blocks[-1][-1], dec_hid_dim=dec_hid_dim)

    def forward(self, supports, inputs):
        """
        :param supports: list of pre-computed adjacency matrix
        :param inputs: [batch_size, seq_len, num_nodes, d_in]
        :return: outputs: tensor. [batch_size, seq_len, num_nodes, dec_hid_dim]
                 hidden: tensor. [batch_size, num_nodes, dec_hid_dim]
        """
        outputs = inputs
        if self.adpt_type == "pam":
            feature_data = inputs[..., 0].squeeze()
            adpt_adj = self.getpam(feature_data)
        elif self.adpt_type == "random_embedding":
            adpt_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        elif self.adpt_type == "no_adpt":
            adpt_adj = None
        else:
            raise ValueError("adpt_adj should be one of {0}, {1} and {2}".format("pam", "random_embedding", "no_adpt"))
        for i, _ in enumerate(self.blocks):
            outputs = self.st_conv_blocks[i](supports, outputs, adpt_adj)

        # Output Layer maps outputs to the initial hidden state for the decoder.
        # this can be seen as aggregating all the source information to a context vector, and
        # use it as initial hidden states for the decoder.
        hidden = self.output_layer(outputs)
        return outputs, hidden
