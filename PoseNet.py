import torch
import torch.nn as nn
from BiLSTM import BiLSTMLayer
import Module
import math

class PoseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, device=torch.device("cuda:0")):
        super(PoseNet, self).__init__()

        self.device = device
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        self.probs_log = []

        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = num_classes

        self.conv1d = Module.TemporalConv(input_size=self.input_size,
                                           hidden_size=self.hidden_size,
                                           conv_type=2)

        self.conv1d1 = Module.TemporalConv(input_size=self.input_size,
                                           hidden_size=self.hidden_size,
                                           conv_type=2)
        
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)

        self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, bidirectional=True)

        self.classifier11 = Module.NormLinear(self.hidden_size, self.output_size)
        self.classifier22 = self.classifier11

        self.classifier33 = Module.NormLinear(self.hidden_size, self.output_size)
        self.classifier44 = self.classifier33

        self.classifier55 = Module.NormLinear(self.hidden_size, self.output_size)

        self.reLU = nn.ReLU(inplace=True)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen=None, isTrain=True):
        outData1 = None
        outData2 = None
        outData3 = None
        logProbs1 = None
        logProbs2 = None
        logProbs3 = None
        logProbs4 = None
        logProbs5 = None
        
        # seqData: [B, T, V, C]
        # view: [B, T, V*C]; transpose: [B, V*C, T] for conv1d
        batch, temp, _, _ = seqData.shape

        framewise = seqData.view(batch, temp, -1).transpose(1, 2)

        framewise1 = framewise.transpose(1, 2).float()
        X = torch.fft.fft(framewise1, dim=-1, norm="forward")
        X = torch.abs(X)
        framewise1 = X.transpose(1, 2)

        conv1d_outputs = self.conv1d(framewise, dataLen)
        x = conv1d_outputs['visual_feat']  # [B, C, T]
        lgt = conv1d_outputs['feat_len']   # list of lengths
        x = x.permute(2, 0, 1)             # to [T, B, C]

        conv1d_outputs1 = self.conv1d1(framewise1, dataLen)
        x1 = conv1d_outputs1['visual_feat']  # [B, C, T]
        x1 = x1.permute(2, 0, 1)             # to [T, B, C]

        lgt_tensor = torch.tensor(lgt, device=x.device)
        outputs = self.temporal_model(x, lgt_tensor.cpu())
        outputs1 = self.temporal_model1(x1, lgt_tensor.cpu())


        # Align temporal dimensions before classification
        t_x = x.size(0)
        t_o = outputs['predictions'].size(0)
        min_len = min(t_x, t_o)

        logProbs1 = self.classifier11(outputs['predictions'][:min_len, :, :])

        logProbs2 = self.classifier22(x[:min_len, :, :])

        t_x1 = x1.size(0)
        t_o1 = outputs1['predictions'].size(0)
        min_len1 = min(t_x1, t_o1)

        logProbs3 = self.classifier33(outputs1['predictions'][:min_len1, :, :])

        logProbs4 = self.classifier44(x1[:min_len1, :, :])

        # Align for the combined prediction
        t_o = outputs['predictions'].size(0)
        t_o1 = outputs1['predictions'].size(0)
        min_len_comb = min(t_o, t_o1)
        x2 = outputs['predictions'][:min_len_comb, :, :] + outputs1['predictions'][:min_len_comb, :, :]
        logProbs5 = self.classifier55(x2)

        if not isTrain:
            logProbs1 = logProbs5

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt_tensor, outData1, outData2, outData3

class PoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return self.log_softmax(logits)
    
class PoseTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,  # Important: keeps (B, T, D)
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0.0)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _get_sinusoid_encoding_table(self, seq_len, dim, device):
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)  # (T, 1)
            div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))  # (D/2,)
            pe = torch.zeros(seq_len, dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe  # (T, D)

    # posenet
    # def forward(self, x, lengths):
    #     # x: (B, T, D)
    #     B, T, _ = x.size()
    #     mask = self._lengths_to_mask(lengths, T)  # shape: (B, T)

    #     x = self.input_proj(x)  # (B, T, hidden_dim)

    #     # Add positional encoding (sinusoidal, dynamic)
    #     pos_encoding = self._get_sinusoid_encoding_table(T, x.size(-1), x.device)  # (T, D)
    #     x = x + pos_encoding.unsqueeze(0) 
    #     # x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]
    #     x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, hidden_dim)
    #     logits = self.fc(x)  # (B, T, num_classes)
        
    #     return self.log_softmax(logits), lengths

    # def _lengths_to_mask(self, lengths, max_len):
    #     # lengths: (B,), returns mask: (B, max_len), True for padding positions
    #     mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    #     return mask.to(lengths.device)  # Ensure mask is on the same device
    

    # transformer
    def forward(self, x, lengths):
        # x: (B, T, D)
        B, T, _ = x.size()
        
        # Ensure lengths is a tensor on the same device as x
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=x.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)
            
        mask = self._lengths_to_mask(lengths, T)  # shape: (B, T)

        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Add positional encoding (sinusoidal, dynamic)
        pos_encoding = self._get_sinusoid_encoding_table(T, x.size(-1), x.device)  # (T, D)
        x = x + pos_encoding.unsqueeze(0) 
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Ensure mask is boolean and on the correct device
        mask = mask.to(device=x.device, dtype=torch.bool)
        
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, hidden_dim)
        x = self.final_norm(x)
        logits = self.fc(x)  # (B, T, num_classes)
        
        return self.log_softmax(logits)

    def _lengths_to_mask(self, lengths, max_len):
        # lengths: tensor on device, returns mask: (B, max_len), True for padding positions
        device = lengths.device
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return mask

    def _get_sinusoid_encoding_table(self, seq_len, dim, device):
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe 