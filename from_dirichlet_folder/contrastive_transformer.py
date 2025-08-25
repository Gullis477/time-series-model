import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class ContrastiveTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        input_length,
        d_model=128,
        nhead=8,
        num_layers=6,
        d_rep=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_rep),
        )

    def forward(self, x, padding_mask=None):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.dropout(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        if padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits
