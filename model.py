import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Lägger till positionell information till input-sekvensen, vilket gör
    att transformern kan förstå ordningen på tidsseriedata.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Skapar den positionella inbäddningen med sinus- och cosinusvågor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 'pe' är inte en del av modellens inlärningsbara parametrar
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lägger till den positionella informationen till input-tensor
        return x + self.pe[:, : x.size(1)]


class ContrastiveTransformer(nn.Module):
    """
    En Transformer Encoder för att lära sig representationer av tidsserier
    med hjälp av kontrastiv inlärning.
    """

    def __init__(
        self,
        num_features: int,
        input_length: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transformerar inputdata till modellens dimension (d_model)
        self.input_proj = nn.Linear(num_features, d_model)

        # En speciell token som representerar hela sekvensen
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Lägger till positionsinformation
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length + 1)

        # Bygger transformer-modulen
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Sparar batch-storleken
        B = x.size(0)

        # 1. Transformera input
        x = self.dropout(self.input_proj(x))

        # 2. Lägg till [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Lägg till positionell inbäddning
        x = self.pos_encoder(x)

        # 4. Anpassa paddingmasken för [CLS] token
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        # 5. Skicka datan genom transformern
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # 6. Returnera representationen från [CLS] token
        cls_output = x[:, 0]

        cls_output = F.normalize(cls_output, dim=1)

        return cls_output
