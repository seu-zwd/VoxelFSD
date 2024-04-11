import torch
import math
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_


class MLP(nn.Module):
    def __init__(self, in_features, hide_features, out_features, act_layer=nn.GELU(),
                drop=0., bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hide_features, bias)
        self.act1 = act_layer
        self.drop1 = nn.Dropout(drop)
        # self.norm = norm_layer if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hide_features, out_features, bias)

    def forward(self, x):
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, q, k, v, mask):
        h_dim = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(h_dim)
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9) 
        scores = scores.softmax(dim=-1)
        atten = torch.matmul(scores, v)
        # atten = self.attn_drop(atten)
        return atten, scores

class MultiheadAttention(nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.h = h
        self.att = Attention()

    def forward(self, q, k, v, mask=None):
        B, N, C = q.size()
        assert C % self.h == 0
        q = q.view(B, -1, self.h, C//self.h).transpose(1, 2)
        k = k.view(B, -1, self.h, C//self.h).transpose(1, 2)
        v = v.view(B, -1, self.h, C//self.h).transpose(1, 2)
        atten, score = self.att(q, k, v, mask)    # (B, N, self.H, C//self.h)
        atten = atten.transpose(1, 2).contiguous().view(B, N, -1)
        return atten

class DecoderBlock(nn.Module):
    def __init__(self, h, in_channels, hide_channels, out_channels, act_layer=nn.GELU(), norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True) -> None:
        super().__init__()
        self.self_att = MultiheadAttention(h)
        self.cross_att = MultiheadAttention(h)
        self.mlp = MLP(in_channels, hide_channels, out_channels, act_layer, drop, bias)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.norm3 = norm_layer(in_channels) if norm_layer else nn.Identity()

    def forward(self, q, k, v, mask=None):
        q2 = self.self_att(q, q, q, mask)
        q2 = self.norm1(q + self.dropout1(q2))
 
        q3 = self.cross_att(q2, k, v, mask)
        q3 = self.norm2(q2 + self.dropout2(q3))
        out = self.mlp(q3)
        out = self.norm3(q3+self.dropout3(out))
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, h, in_channels, hide_channels, out_channels, act_layer=nn.GELU(), norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True) -> None:
        super().__init__()
        # self.pe = PositionalEncoding(in_channels)
        self.att = MultiheadAttention(h)
        self.mlp = MLP(in_channels, hide_channels, out_channels, act_layer, drop, bias)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q, k, v, mask=None):
        q2 = self.att(q, k, v, mask)
        q2 = self.norm1(q + self.dropout1(q2))
        out = self.mlp(q2)
        out = self.norm2(q2+self.dropout2(out))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, cfg,
                  act_layer=nn.GELU(), norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0.1, bias=True) -> None:
        super().__init__()
        N = cfg.NUM_LAYERS
        H = cfg.NUM_HEADS
        r = cfg.RATIO
        d_model = cfg.IN_FEATURES
        self.pos_dim = cfg.POS_DIM

        # self.pos_en = PositionalEncoding(d_model)

        self.Q_linear = nn.Linear(d_model, d_model, bias=False)
        self.K_linear = nn.Linear(d_model, d_model, bias=False)
        self.V_linear = nn.Linear(d_model, d_model, bias=False)

        self.blocks = nn.ModuleList([EncoderLayer(H, d_model, d_model*r, d_model,
                                           act_layer, norm_layer, drop, bias) for i in range(N)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q_input, mask=None):
        # q_input = torch.cat([q, pos], -1)
        q = self.Q_linear(q_input)
        k = self.K_linear(q_input)
        v = self.V_linear(q_input)
        # q = self.pos_en(q)

        for i, block in enumerate(self.blocks):
            q = block(q, k, v, mask)
        return q

class TransformerDecoder(nn.Module):
    def __init__(self, cfg,
                  act_layer=nn.GELU(), norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0.1, bias=True) -> None:
        super().__init__()
        N = cfg.NUM_LAYERS
        H = cfg.NUM_HEADS
        r = cfg.RATIO
        d_model = cfg.IN_FEATURES
        self.pos_dim = cfg.POS_DIM
        # self.pos_en = PositionalEncoding(d_model)
        self.Q_linear = nn.Linear(d_model, d_model, bias=False)
        self.K_linear = nn.Linear(d_model, d_model, bias=False)
        self.V_linear = nn.Linear(d_model, d_model, bias=False)
        self.blocks = nn.ModuleList([DecoderBlock(H, d_model, d_model*r, d_model,
                                           act_layer, norm_layer, drop, bias) for i in range(N)])

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q_input, k_input, v_input, mask=None):
        # q_input = torch.cat([q, pos], -1)
        # k_input = torch.cat([k, pos], -1)
        # v_input = torch.cat([v, pos], -1)
        q = self.Q_linear(q_input)
        k = self.K_linear(k_input)
        v = self.V_linear(v_input)
        # q = self.pos_en(q)
        for i, block in enumerate(self.blocks):
            q = block(q, k, v, mask)
        return q


class AFB(nn.Module):
    def __init__(self, model_cfg):
        super(AFB, self).__init__()
        self.pos_en = nn.Linear(7, model_cfg.TRANSFORMER_CFG.IN_FEATURES)
        self.encoder = TransformerEncoder(model_cfg.TRANSFORMER_CFG)
        self.decoder = TransformerDecoder(model_cfg.TRANSFORMER_CFG)

    def forward(self, q, kv, pos, mask=None):
        pe = self.pos_en(pos)
        q = q + pe
        kv = kv + pe
        q = self.encoder(q)
        out = self.decoder(kv, q, q)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class BoxPositionalEncoding(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(7, planes//2, 1),
            nn.BatchNorm1d(planes//2),
            nn.ReLU(),
            nn.Conv1d(planes//2, planes, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ffn(x)
        x = x.transpose(1, 2)
        return x

if __name__=="__main__":
    input_tensor = torch.randn((4, 32, 128))  # (batch_size, sequence_length, embed_dim)
    # # MHA = MultiheadAttention(4)
    # # output = MHA(input_tensor, input_tensor, input_tensor)
    # block = DecoderBlock(4, 128, 256, 128)
    # output = block(input_tensor, input_tensor, input_tensor)

    # box = torch.randn((4, 32, 7))
    # posencoder = BoxPositionalEncoding()
    # embeding = posencoder(box)
    # print(output.size(), embeding.size())
    PE = PositionalEncoding(128)
    out = PE(input_tensor)
    print(out.size())
