# src/models/lstm_weather.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, n_features, hidden_size, out_len, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_features)
        self.out_len = out_len

    def forward(self, decoder_input, hidden, teacher_forcing_inputs=None):
        outputs = []
        h, c = hidden
        seq_input = decoder_input
        for t in range(self.out_len):
            out, (h, c) = self.lstm(seq_input, (h, c))
            pred = self.fc(out.squeeze(1))
            outputs.append(pred.unsqueeze(1))
            if teacher_forcing_inputs is not None:
                seq_input = teacher_forcing_inputs[:, t:t+1, :].float()
            else:
                seq_input = pred.unsqueeze(1).float()
        return torch.cat(outputs, dim=1)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=128, out_len=15, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(n_features, hidden_size, out_len, num_layers, dropout)

    def forward(self, x, teacher_forcing_inputs=None):
        h, c = self.encoder(x)
        decoder_init = x[:, -1:, :]
        out = self.decoder(decoder_init, (h, c), teacher_forcing_inputs=teacher_forcing_inputs)
        return out
