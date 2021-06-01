import math
import copy
import time
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat


MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class ResidualAttn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y, z, **kwargs):
        return self.fn(x, y, z, **kwargs) + x


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, z, **kwargs):
        return self.fn(self.norm(x), self.norm(y), self.norm(z), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, mask = None):
        b, n, _, h = *query.shape, self.heads
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        qkv = (q, k, v)
        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            """
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            print(mask.shape)
            print(dots.shape)
            print(mask)
            dots.masked_fill_(~mask, mask_value)
            del mask
            """
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -1e9)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualAttn(PreNormAttn(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, x, x, mask = mask)
            x = ff(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualAttn(PreNormAttn(dim, Attention(dim, heads = heads, dim_head = dim_head, 
                                                                    dropout = dropout))),
                ResidualAttn(PreNormAttn(dim, Attention(dim, heads = heads, dim_head = dim_head, 
                                                                    dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, memory, src_mask=None, target_mask=None):
        m = memory
        for self_attn, src_attn, ff in self.layers:
            x = self_attn(x, x, x, mask=target_mask)
            x = src_attn(x, m, m, mask=src_mask)
            x = ff(x)
        return x


class TansformerAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, dim, depth, heads, 
                                dim_head, mlp_dim, dropout):
        super().__init__()
        self.encoder = encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder = decoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def encode(self, src_emb, src_mask=None):
        return self.encoder(src_emb, src_mask)
    
    def decode(self, target_emb, memory, src_mask=None, target_mask=None):
        return self.decoder(target_emb, memory, src_mask, target_mask)
    
    def forward(self, src_emb, target_emb, src_mask=None, target_mask=None):
        return self.decode(target_emb, self.encode(src_emb, src_mask), src_mask, target_mask)



class SAAE(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, in_channels, latent_dim, is_bn=True, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.cnn_encoder = nn.Sequential(*layers)

        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.input_patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder = Decoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder_input_emb = nn.Parameter(torch.rand(1,num_patches,dim))
        self.to_latent = nn.Identity()
        self.embedding_to_patch = nn.Linear(dim, patch_dim)

        layers2 = []
        layers2 += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers2 += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers2 += [nn.ReLU()]
        layers2 += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers2 += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers2 += [nn.ReLU()]
        layers2 += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]

        self.cnn_decoder = nn.Sequential(*layers2)

    def forward(self, img, mask = None):
        p = self.patch_size

        # dimension reduction
        x = self.cnn_encoder(img)

        # split feature map into patches
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.input_patch_to_embedding(x)

        # b: batch size, n: num of patches, _: hidden dim
        b, n, _ = x.shape
        # add position encoding to embedding
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # encode
        memory = self.encoder(x)

        decoder_input_emb = repeat(self.decoder_input_emb, '() n d -> b n d', b=b)
        
        decoder_input_emb += self.pos_embedding[:, :n]

        x = self.decoder(decoder_input_emb, memory)
        
        # 将embedding转换成patch
        x = self.embedding_to_patch(x)
        # 将patch组成完整图像
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = p, p2 = p, 
                                                                 h=img.size()[2]//p)

        x = self.cnn_decoder(x)
        return x
    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss

if __name__ == "__main__":
    b = SAAE(
        image_size = 64,
        patch_size = 8,
        dim = 512,
        depth = 3,
        heads = 4,
        mlp_dim = 256,
        in_channels = 3456,
        latent_dim=100,
        is_bn=True,
        channels = 100,
        dropout = 0.0,
        emb_dropout = 0.0
    )

    """
    for p in b.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    """

    img = torch.randn(2,3456,64,64)

    x = b(img)
    criterion = nn.MSELoss()
    print(criterion(x, img))

