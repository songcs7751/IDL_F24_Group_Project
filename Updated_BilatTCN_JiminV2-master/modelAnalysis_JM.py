import torch
import torch.nn as nn
import biomechdata as n

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = n.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = n.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BilatTCN(nn.Module):
    def __init__(self, input_size_c, input_size_nc, output_size, num_channels_c, num_channels_nc, ksize_c, ksize_nc, dropout, eff_hist_c, eff_hist_nc):
        super(BilatTCN, self).__init__()
        self.tcn_causal = TemporalConvNet(input_size_c, num_channels_c, kernel_size=ksize_c, dropout=dropout)
        self.tcn_noncausal = TemporalConvNet(input_size_nc, num_channels_nc, kernel_size=ksize_nc, dropout=dropout)
        self.linear = nn.Linear(num_channels_c[-1] + num_channels_nc[-1], output_size)
        self.init_weights()
        self.eff_hist_c = eff_hist_c
        self.eff_hist_nc = eff_hist_nc

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, sequence_lens=[]):
        x_c = x[:, :, :-self.eff_hist_nc]
        x_nc = x[:, :, self.eff_hist_c:]
        x_nc = torch.flip(x_nc, (2,))
        x_c = self.tcn_causal(x_c)
        x_nc = self.tcn_noncausal(x_nc)

        if any(sequence_lens):
            y1_c = torch.cat([x_c[i, :, self.eff_hist_c:self.eff_hist_c+sequence_lens[i]].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
            y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:self.eff_hist_nc+sequence_lens[i]].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()
        else:
            y1_c = torch.cat([x_c[i, :, self.eff_hist_c:].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
            y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()

        y1 = torch.cat((y1_c, y1_nc), dim=1).contiguous()
        return self.linear(y1)

# Load the model checkpoint
model_path = '/Users/sunho/Desktop/TCN/Output/SavedModels/AB05_dropout0.3_hsize50_ksize_c4_ksize_nc4_levels_c5_levels_nc5_lossMSELoss_lr0.0005_optAdam_pred0.tar'
checkpoint = torch.load(model_path)

# Extract model parameters
model_params = {
    'input_size_c': checkpoint['input_size_c'],
    'input_size_nc': checkpoint['input_size_nc'],
    'output_size': checkpoint['output_size'],
    'num_channels_c': checkpoint['num_channels_c'],
    'num_channels_nc': checkpoint['num_channels_nc'],
    'ksize_c': checkpoint['ksize_c'],
    'ksize_nc': checkpoint['ksize_nc'],
    'dropout': checkpoint['dropout'],
    'eff_hist_c': checkpoint['eff_hist_c'],
    'eff_hist_nc': checkpoint['eff_hist_nc'],
}

# Create the model
model = BilatTCN(**model_params)

# Load the state_dict
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Print the model structure
print(model)

