# src/models/lstm_weather.py
import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=128,
        out_len=15,
        num_layers=2,
        dropout=0.15,
        use_attention=False,
    ):
        super().__init__()
        self.out_len = out_len
        self.use_attention = use_attention

        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.decoder_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.out_proj = nn.Linear(hidden_size, n_features)
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attn = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        hx, cx = h[-1], c[-1]

        dec_in = self.input_proj(x.mean(dim=1))
        outputs = []

        for _ in range(self.out_len):
            hx, cx = self.decoder_cell(self.dropout(dec_in), (hx, cx))
            if self.use_attention:
                ctx = enc_out.mean(dim=1)
                hx = hx + self.attn(torch.cat([hx, ctx], dim=1))
            out = self.out_proj(hx)
            outputs.append(out.unsqueeze(1))
            dec_in = out

        return torch.cat(outputs, dim=1)
