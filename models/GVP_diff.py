import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from .utils import register_model
from .esm_block import DihedralFeatures
from .transformer_layer import TransformerEncoderCondLayer, SinusoidalPositionalEmbedding


@torch.no_grad()
def geo_batch(batch):
    data_list = []
    # print(len(batch['z_t']))
    batch_size, length = batch['z_t'].shape[:2]

    for i in range(batch_size):
        data_list.append(torch_geometric.data.Data(
            z_t=batch['z_t'][i],
            seq=batch['seq'][i],  # num_res x 1
            coords=batch['coords'][i],  # num_res x 3 x 3
            node_s=batch['node_s'][i],  # num_res x num_conf x 4
            node_v=batch['node_v'][i],  # num_res x num_conf x 4 x 3
            edge_s=batch['edge_s'][i],  # num_edges x num_conf x 32
            edge_v=batch['edge_v'][i],  # num_edges x num_conf x 1 x 3
            edge_index=batch['edge_index'][i],  # 2 x num_edges
            mask=batch['mask'][i]  # num_res x 1
        ))

    return Batch.from_data_list(data_list), batch_size, length

@register_model(name='GVPTransCond')
class GVPTransCond(torch.nn.Module):
    '''
    GVP + Transformer model for RNA design

    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in encoder/decoder
    :param drop_rate: rate to use in all dropout layers
    :param out_dim: output dimension (4 bases)
    '''

    def __init__(self, config):
        super().__init__()
        self.node_in_dim = tuple(config.model.node_in_dim)  # node_in_dim
        self.node_h_dim = tuple(config.model.node_h_dim)  # node_h_dim
        self.edge_in_dim = tuple(config.model.edge_in_dim)  # edge_in_dim
        self.edge_h_dim = tuple(config.model.edge_in_dim)  # edge_h_dim
        self.num_layers = config.model.num_layers
        self.out_dim = config.model.out_dim
        self.time_cond = config.model.time_cond
        self.dihedral_angle = config.model.dihedral_angle
        self.drop_struct = config.model.drop_struct
        drop_rate = config.model.drop_rate
        activations = (F.relu, None)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                         activations=activations, vector_gate=True,
                         drop_rate=drop_rate)
            for _ in range(self.num_layers))

        # Output
        self.W_out = GVP(self.node_h_dim, (self.node_h_dim[0], 0), activations=(None, None))

        # Transformer Layers
        self.seq_res = nn.Linear(self.node_in_dim[0], self.node_h_dim[0])
        self.mix_lin = nn.Linear(self.node_h_dim[0] * 2, self.node_h_dim[0])
        self.num_trans_layer = config.model.num_trans_layer
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.node_h_dim[0],
            -1,
        )
        self.trans_layers = nn.ModuleList(
            TransformerEncoderCondLayer(config.model.trans)
            for _ in range(self.num_trans_layer))
        self.MLP_out = nn.Sequential(
            nn.Linear(self.node_h_dim[0], self.node_h_dim[0]),
            nn.ReLU(),
            nn.Linear(self.node_h_dim[0], self.out_dim)
        )

        # Time conditioning
        if self.time_cond:
            learned_sinu_pos_emb_dim = 16
            time_cond_dim = config.model.node_h_dim[0] * 2
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
            self.to_time_hiddens = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, config.model.node_h_dim[0]),
            )

        # Dihedral angle
        if self.dihedral_angle:
            self.embed_dihedral = DihedralFeatures(config.model.node_h_dim[0])

    def struct_forward(self, batch, init_seq, batch_size, length, **kwargs):
        h_V = (init_seq, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        if self.dihedral_angle:
            dihedral_feats = self.embed_dihedral(batch.coords).reshape_as(h_V[0])
            h_V = (h_V[0] + dihedral_feats, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        gvp_output = self.W_out(h_V).reshape(batch_size, length, -1)
        return gvp_output

    def forward(self, batch, cond_drop_prob=0., **kwargs):
        # construct extra node and edge features
        batch, batch_size, length = geo_batch(batch)

        z_t = batch.z_t
        cond_x = kwargs.get('cond_x', None)
        if cond_x is None:
            cond_x = torch.zeros_like(batch.z_t)
        else:
            cond_x = cond_x.reshape_as(batch.z_t)

        init_seq = torch.cat([z_t, cond_x], -1)

        if self.training:
            if self.drop_struct > 0 and random.random() < self.drop_struct:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                gvp_output = self.struct_forward(batch, init_seq, batch_size, length, **kwargs)
        else:
            if cond_drop_prob == 0.:
                gvp_output = self.struct_forward(batch, init_seq, batch_size, length, **kwargs)
            elif cond_drop_prob == 1.:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                raise ValueError(f'Invalid cond_drop_prob: {cond_drop_prob}')

        trans_x = torch.cat([gvp_output, self.seq_res(init_seq.reshape(batch_size, length, -1))], dim=-1)
        trans_x = self.mix_lin(trans_x)

        if self.time_cond:
            noise_level = kwargs.get('noise_level')
            time_cond = self.to_time_hiddens(noise_level)  # [B, d_s]
            time_cond = time_cond.unsqueeze(1).repeat(1, length, 1)  # [B, length, d_s]
        else:
            time_cond = None

        # add position embedding
        seq_mask = torch.ones((batch_size, length), device=batch.z_t.device)
        pos_emb = self.embed_positions(seq_mask)

        trans_x = trans_x + pos_emb
        trans_x = trans_x.transpose(0, 1)

        # transformer layers
        for layer in self.trans_layers:
            trans_x = layer(trans_x, None, cond=time_cond.transpose(0, 1))

        logits = self.MLP_out(trans_x.transpose(0, 1))
        # logits = logits.reshape(batch_size, -1, self.out_dim)
        return logits


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x = rearrange(x, 'b -> b 1')
        x = x.unsqueeze(-1)
        # freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


#########################################################################

class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(
            self,
            node_dims,
            edge_dims,
            n_message=3,
            n_feedforward=2,
            drop_rate=.1,
            autoregressive=False,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
            residual=True
    ):

        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=x.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


#########################################################################

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = x.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)

