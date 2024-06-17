import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class SymbolicEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_channels = 3
        patch_size = config.patch_size
        symbol_max_seq_length = config.symbol_max_seq_length
        image_size = (patch_size, patch_size * symbol_max_seq_length)
        embed_dim = config.embed_dim

        image_size = to_2tuple(image_size, )
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, symb_values):
        batch_size, num_channels, height, width = symb_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        E_symb = self.projection(symb_values).flatten(2).transpose(1, 2)
        return E_symb


def get_experts(num_experts, embed_dim):
    experts = nn.Sequential()
    for i in range(num_experts):
        experts.append(nn.Embedding(embed_dim, embed_dim))
    return experts


class SelectiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.k = config.num_selected_experts
        self.gate = nn.Linear(config.embed_dim, self.num_experts)
        self.experts = get_experts(self.num_experts, config.embed_dim)

    def forward(self, E_symb):
        batch_size, seq_length, embed_dim = E_symb.shape

        # Router
        hidden_states = E_symb.reshape(-1, embed_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Experts
        E_bias = torch.zeros((batch_size * seq_length, embed_dim), dtype=hidden_states.dtype,
                             device=hidden_states.device)

        # Create a mask for each expert
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop through each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # Find the locations where the current expert is selected
            top_x, id_x = torch.where(expert_mask[expert_idx])

            # If there are no selected locations, skip this expert
            if id_x.shape[0] == 0:
                continue

            # Get the current hidden state for this expert
            current_state = hidden_states[None, id_x].reshape(-1, embed_dim)

            # Calculate the hidden state for this expert
            current_hidden_states = torch.matmul(current_state, expert_layer.weight.data.cuda())

            # Add the hidden state to the bias_embeddings tensor
            E_bias.index_add_(0, id_x, current_hidden_states.to(hidden_states.dtype))

        E_bias = E_bias.reshape(batch_size, seq_length, embed_dim)
        return E_bias


def transformer_encoder(num_encoder_layers, dropout, embed_dim):
    encoder_layer = TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=2048, dropout=dropout)
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
    return encoder


def transformer_decoder(num_decoder_layers, dropout, embed_dim):
    decoder_layer = TransformerDecoderLayer(embed_dim, nhead=8, dim_feedforward=2048, dropout=dropout)
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder


class SpatialEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_encoder_layers = config.encoder_layers
        num_decoder_layers = config.decoder_layers
        dropout = config.dropout_rate
        embed_dim = config.embed_dim

        self.encoder = transformer_encoder(num_encoder_layers, dropout, embed_dim)
        self.decoder = transformer_decoder(num_decoder_layers, dropout, embed_dim)

    def forward(self, E_meta):
        E_h = self.translate_encoder(E_meta)
        E_g = self.translate_decoder(E_h, E_meta)
        return E_h, E_g


class SSSEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.symbolic_embedding = SymbolicEmbedding(config)

        self.selective_embedding = SelectiveEmbedding(config)

        self.spatial_embedding = SpatialEmbedding(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # cls_token and position_embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.symbolic_embedding.num_patches + 1, config.embed_dim), requires_grad=False)
        self.initialize_weights()

    def forward(self, symb_values, attention_mask=None):
        batch_size, num_channels, height, width = symb_values.shape

        E_symb = self.symbolic_embedding(symb_values)
        E_symb = F.softmax(E_symb, dim=2)

        E_bias = self.selective_embedding(E_symb)
        E_meta = E_symb + E_bias

        # add position embeddings w/o cls token
        E_meta = E_meta + self.position_embeddings[:, 1:, :]

        # get cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(E_meta.shape[0], -1, -1)

        # append cls token
        E_meta = torch.cat((cls_tokens, E_meta), dim=1)
        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        E_meta = self.LayerNorm(E_meta)
        E_meta = self.dropout(E_meta)
        E_h, E_g = self.transfer_Embeddings(E_meta)
        return E_h, E_g, attention_mask

    def initialize_weights(self):
        # initialize and freeze position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.symbolic_embedding.num_patches ** 0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize symbolic_embedding
        w = self.symbolic_embedding.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return x, x


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
