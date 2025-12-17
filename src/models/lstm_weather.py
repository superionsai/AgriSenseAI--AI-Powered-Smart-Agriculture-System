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
        """
        Seq-to-seq LSTM with a simple LSTMCell decoder.
        - n_features: number of input features (and output features)
        - hidden_size: latent size of LSTM
        - out_len: sequence length to predict
        """
        super().__init__()
        self.n_features = n_features
        self.out_len = out_len
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        # Encoder: consumes (batch, seq_len, n_features)
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder: LSTMCell operates on hidden vectors of size hidden_size
        self.decoder_cell = nn.LSTMCell(hidden_size, hidden_size)

        # Project input features -> hidden_size for decoder start and for feeding decoder
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Project hidden -> output features
        self.out_proj = nn.Linear(hidden_size, n_features)

        self.dropout = nn.Dropout(dropout)

        if use_attention:
            # simple additive-style projection (kept small & explainable)
            self.attn = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        x: (batch, in_len, n_features)
        returns: (batch, out_len, n_features)
        """
        # Basic input validation for helpful errors
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, F), got {tuple(x.shape)}")
        if x.size(2) != self.n_features:
            raise ValueError(f"Input has {x.size(2)} features but model expects {self.n_features}")

        # Encoder
        enc_out, (h, c) = self.encoder(x)
        # Use last layer's last hidden state as initial decoder hidden state
        hx = h[-1]  # (batch, hidden)
        cx = c[-1]  # (batch, hidden)

        # Initialize decoder input by projecting last encoder time-step (common practice)
        # x[:, -1, :] shape -> (batch, n_features); input_proj -> (batch, hidden_size)
        dec_in = self.input_proj(x[:, -1, :])

        outputs = []
        for _ in range(self.out_len):
            # dec_in is always of hidden_size (we ensure via input_proj below)
            hx, cx = self.decoder_cell(self.dropout(dec_in), (hx, cx))

            if self.use_attention:
                # simple context: mean over encoder outputs
                ctx = enc_out.mean(dim=1)  # (batch, hidden)
                att_in = torch.cat([hx, ctx], dim=1)
                att = torch.tanh(self.attn(att_in))
                hx = hx + att

            out = self.out_proj(hx)  # (batch, n_features)
            outputs.append(out.unsqueeze(1))

            # Project the produced output back into hidden space for next step
            dec_in = self.input_proj(out)

        out_seq = torch.cat(outputs, dim=1)  # (batch, out_len, n_features)
        return out_seq
