import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from os import listdir
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from biomechutils import get_file_names

class AbleBodyDataset2(Dataset):
	def __init__(self, input_dir, label, ft_use=[], ft_ignore=[], trials_use=[], trials_ignore=[], standardize=False):
		self.input_dir = input_dir
		self.label = label
		self.ft_use = ft_use
		self.ft_ignore = ft_ignore
		self.trials_use = trials_use
		self.trials_ignore = trials_ignore
		self.trial_names = []
		self.standardize = standardize
		print('Initializing AbleBodyDataset2')
		print(f"Input directory: {self.input_dir}")
		print(f"Label: {self.label}")
		print(f"Features to use: {self.ft_use}")
		print(f"Features to ignore: {self.ft_ignore}")
		print(f"Trials to use: {self.trials_use}")
		print(f"Trials to ignore: {self.trials_ignore}")
		print(f"Standardize: {self.standardize}")


	def __len__(self):
		return len(self.trial_names)

	def __getitem__(self, idx):
		# print("in __getitem__")
		# print(f"idx: {idx}")
		# print(f"trial_names: {self.trial_names}")
		if isinstance(idx, list):
			trial_names = [self.trial_names[i] for i in idx]
		else:
			trial_names = self.trial_names[idx]
			if not isinstance(trial_names, list):
				trial_names = [trial_names]

		input_data = [self.read_tensor(self.input_dir + '/' + name) for name in trial_names] # 이건 맞으니까 건들 ㄴㄴ
		# print(f"input_data: {type(input_data), type(input_data[0])}")
		# ProcessedData 후에 \ 더하고 AB01, AB02 박는 역할
		label_data = tuple([data[1] for data in input_data])
		label_data = torch.cat(label_data, dim=0)
		# print(f"label_data: {label_data.shape}")
		feature_data = tuple([data[0] for data in input_data])
		feature_data = torch.cat(feature_data, dim=0)
		# print("feature_data: ", feature_data.shape)

		return {'features': feature_data, 'labels': label_data}


	def add_trials(self, sub_dir='', ext='', search=True, include=[]):
		#print(f"Kanye: {self.input_dir}")
		input_dir = self.input_dir + '/' + sub_dir # 없으면 안되는 코드
		print(f"add_trials: {input_dir}")
		print(f"search: {search}")
		print(f"include: {include}")
		if any(include):
			search_dirs = include
		elif search:
			search_dirs = listdir(input_dir)
			# print(f"search_dirs: {search_dirs}")
		else:
			search_dirs = []


		trial_names = get_file_names(input_dir, sub_dir=search_dirs, ext=ext)
		# trial_names = get_file_names(input_dir, ext=ext)
		# print(f"Kendrik: {trial_names}")		
		trial_names = [sub_dir + '/' + t for t in trial_names]
		# new_trial_names = []  # 새로운 리스트를 생성
		# for t in trial_names:  # 기존 trial_names 리스트를 순회
		# 	jimin_dir = sub_dir + '/' + t
		# 	# print(f"jimin_dir: {jimin_dir}")
		# 	new_trial_names.append(jimin_dir)  # 각 요소에 대해 디렉토리 생성 후 추가
		# trial_names = new_trial_names  # 기존 리스트를 업데이트

		if any(self.trials_use):
			trial_names = [t for t in trial_names if any([n for n in self.trials_use if n in t])]
		if any(self.trials_ignore):
			trial_names = [t for t in trial_names if not any([n for n in self.trials_ignore if n in t])]
		self.trial_names += trial_names
		return True


	def get_boundaries(self, data, eff_hist, eff_pred, trial):
		# Find start and end of valid label data (based on padding in preprocessing)
		non_nan = np.where(~np.isnan(data['labels'].numpy()))[0]
		# print(non_nan.shape)
		if non_nan.shape[0] == 0:
			return None, None, None, None

		# print(f"non_nan: {non_nan}")	
		label_start = np.where(~np.isnan(data['labels'].numpy()))[0][0]
		label_end = np.where(~np.isnan(data['labels'].numpy()))[0][-1]

		# Check if there is enough data before start of label for the given effective history and update accordingly
			# label_start = how many data points exist before the first label for this step data.
			# eff_hist = the number of data points required as input to the TCN for the first label
			# eff_pred = the number of data points between the last input and first label for the TCN (prediction)
		if eff_hist+eff_pred > label_start:
			print(f'Warning - Starting later than padded input data. Removing {(eff_hist+eff_pred)-label_start} labels from start of {trial}.')
		label_start = max(eff_hist+eff_pred, label_start)
		feat_start = label_start-(eff_hist+eff_pred)

		# Similar check as before but now we only need to make sure there is enough room for any prediction
			# Technically this only matters for negative prediction values (estimating back in time from the input data)
		if label_end > data['labels'].shape[0] + eff_pred - 1:
			print(f"Warning - Ending earlier than padded input data. Removing {(data['labels'].shape[0] + eff_pred - 1)-label_end} labels from end of {trial}.")
		label_end = min(data['labels'].shape[0]+eff_pred-1, label_end)
		feat_end = label_end-eff_pred

		return feat_start, feat_end, label_start, label_end

	def load_trial_dict(self, eff_hist, eff_pred=0, batch_size_per_trial=1, device='cpu', verbose=False):
		trial_dict = {trial: {} for trial in self.trial_names}
		trial_list = []
		for i, trial in enumerate(self.trial_names):
			if verbose:
				print('Loading ' + trial)
			
			# Get trial data from dataset
			data = self[i]
			# print("data: ", type(data))
			feat_start, feat_end, label_start, label_end = self.get_boundaries(data, eff_hist, eff_pred, trial)
			if feat_start is None:
				print(f"Skipping {trial} due to insufficient data.")
				continue

			# Save input and label data to training dictionary
			trial_dict[trial]['data'] = data['features'][feat_start:feat_end+1, :].float().to(device)
			trial_dict[trial]['labels'] = data['labels'][label_start:label_end+1].float().to(device)

			# Slice trial based on batch_size_per_trial and randomize order of slices
			label_length = trial_dict[trial]['labels'].shape[0]
			num_slices = int(np.ceil(label_length / batch_size_per_trial))
			slice_order = np.arange(num_slices)
			np.random.shuffle(slice_order)
			trial_dict[trial]['slice_order'] = list(slice_order)
			trial_dict[trial]['slice_count'] = 0

			# Add trial to training trial list num_slices times
			trial_list += [trial]*num_slices

			if verbose and i==0:
				print(self.features_use)

		return trial_dict, trial_list

	def load_trial_list(self, eff_hist, eff_pred=0, device='cpu', verbose=False):
		x_list = []
		y_list = []
		y_trial_count = [] # Label for each step through the timeseries data
		count = 0
		for i, trial in enumerate(self.trial_names):
			if verbose:
				print('Loading ' + trial)

			# Get trial data from dataset
			data = self[i]
			feat_start, feat_end, label_start, label_end = self.get_boundaries(data, eff_hist, eff_pred, trial)
			if feat_start is None:
				print(f"Skipping {trial} due to insufficient data.")
				continue

			# Save input and label data to test list
			x_list.append(data['features'][feat_start:feat_end+1, :].float().to(device))
			y_list.append(data['labels'][label_start:label_end+1].float().to(device))

			# Update y_trial_count with the step number for this trial for each timestep in the data
			count += 1
			y_trial_count += [count]*y_list[-1].shape[0]

		return x_list, y_list, y_trial_count

# 	def read_tensor(self, file_path):
# 		print(f"Reading file: {file_path}")
# 		df = pd.read_csv(file_path)  # Read in trial data
# 		print(f"Initial DataFrame shape: {df.shape}")

# 		# Select specified features
# 		features_use = list(df.columns)
# 		if any(self.ft_use):
# 			features_use = [feature for feature in features_use if any([f for f in self.ft_use if f in feature])]
# 		if any(self.ft_ignore):
# 			features_use = [feature for feature in features_use if not any([f for f in self.ft_ignore if f in feature])]
# 		features_use = sorted(features_use)
# 		self.features_use = features_use

# # 		print(f"Selected features: {self.features_use}")
# # 나중에 feature selection 관련해서 다시 발동해서 체크. 왜 순서가 내가 지정한대로가 아닌지?

# 		missing_ft = [f for f in self.ft_use if not any([f for feature in features_use if f in feature])]
# 		if any(missing_ft):
# 			print(f"Missing {(', ').join(missing_ft)} in input data!")

# 		# Convert data to pytorch tensor
# 		input_data = torch.tensor(df[features_use].values)
# 		label_data = torch.tensor([df[self.label].values,]).T
# 		print(f"Input tensor shape: {input_data.shape}")
# 		print(f"Label tensor shape: {label_data.shape}")

# 		if any([any(torch.isnan(d)) for d in input_data]):
# 			print('Warning - NaN in input ' + file_path)


# 		return input_data, label_data

	def read_tensor(self, file_path: str):
		# print(f"Reading file: {file_path}")
		df = pd.read_csv(file_path)  # Read in trial data
		df_label = pd.read_csv(file_path.replace('exo', 'moment'))

		df['hip_flexion_l_moment'] = df_label['hip_moment_l']
		df.ffill(inplace=True)
		df.fillna(0, inplace=True)

		if self.standardize:
			normalized_df=(df-df.mean())/df.std()
		else:
			normalized_df = df

		# print(f"Initial DataFrame shape: {df.shape}")
		
		# Select specified features
		# print("columns available in the exo dataframe: ", df.columns)
		# print("columns available in the moment dataframe: ", df_label.columns)
		features_use = list(df.columns)

		if any(self.ft_use):
			features_use = [feature for feature in features_use if feature in self.ft_use]
		if any(self.ft_ignore):
			features_use = [feature for feature in features_use if feature not in self.ft_ignore]
		features_use = sorted(features_use)
		self.features_use = features_use
		
		# Debugging: Print selected features
		# print(f"Selected features: {self.features_use}")
		
		missing_ft = [f for f in self.ft_use if f not in features_use]
		if any(missing_ft):
			print(f"Missing features in input data: {', '.join(missing_ft)}")

    	# Convert data to pytorch tensor
		input_data = torch.tensor(normalized_df[features_use].values)
		label_data = torch.tensor(df[self.label].values).unsqueeze(1)
		# print(f"Input tensor shape: {input_data.shape}")
		# print(f"Label tensor shape: {label_data.shape}")
		
		if torch.isnan(input_data).any():
			percent_nan = torch.isnan(input_data).sum().item() / input_data.numel()
			print('Warning - {} NaN in input '.format(percent_nan) + file_path)
		
		return input_data, label_data




def get_model_dict(model_type, h_dict):
	if model_type=='BilatTCN':
		return get_tcn_dict(h_dict)

def get_tcn_dict(h_dict):
	m_dict = {k: h_dict[k] for k in ('ksize_c', 'ksize_nc', 'dropout')}
	m_dict['eff_hist_c'] = 2*sum([(h_dict['ksize_c']-1)*(2**level) for level in range(h_dict['levels_c'])])
	m_dict['eff_hist_nc'] = 2*sum([(h_dict['ksize_nc']-1)*(2**level) for level in range(h_dict['levels_nc'])])
	m_dict['num_channels_c'] = [h_dict['hsize']]*h_dict['levels_c']
	m_dict['num_channels_nc'] = [h_dict['hsize']]*h_dict['levels_nc']
	return m_dict

class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()

		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
			stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
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
		# out = self.conv1(x)
		# print(out.shape)
		
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


class TCN(nn.Module):
	def __init__(self, input_size, output_size, num_channels, ksize, dropout, eff_hist, **kwargs):
		super(TCN, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout)
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.init_weights()
		self.eff_hist = eff_hist

	def init_weights(self):
		self.linear.weight.data.normal_(0, 0.01)

	# def forward(self, x, sequence_lens=[], t=-1):
	def forward(self, x, sequence_lens=[]):
		# print(x.shape)
		x = x[:, :,:-self.eff_hist]
		# print('input x', x.shape)
		# print('sequence_lens', sequence_lens)
		y1 = self.tcn(x)
		# start_time = time.time()
		# print('output y1', y1.shape)

		if any(sequence_lens):
			y1 = torch.cat([y1[i, :, self.eff_hist:self.eff_hist+sequence_lens[i]].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()	
		else:
			y1 = torch.cat([y1[i, :, self.eff_hist:].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()
		
		# print('y1 with eff_hist', y1.shape)
		out = self.linear(y1)
		# print('output out', out.shape)
		return out


class BilatTCN(nn.Module):
	def __init__(self, input_size_c, input_size_nc, output_size, num_channels_c, num_channels_nc, ksize_c, ksize_nc, dropout, eff_hist_c, eff_hist_nc):
		super(BilatTCN, self).__init__()
		self.tcn_causal = TemporalConvNet(input_size_c, num_channels_c, kernel_size=ksize_c, dropout=dropout)
		self.tcn_noncausal = TemporalConvNet(input_size_nc, num_channels_nc, kernel_size=ksize_nc, dropout=dropout)
		self.linear = nn.Linear(num_channels_c[-1] + num_channels_nc[-1], output_size)
		self.init_weights()
		self.eff_hist_c = eff_hist_c
		self.eff_hist_nc = eff_hist_nc
		self.num_channels_c = num_channels_c
		self.num_channels_nc = num_channels_nc

	def init_weights(self):
		self.linear.weight.data.normal_(0, 0.01)

	def forward(self, x, sequence_lens=[]):
		x_c = x[:, :,:-self.eff_hist_nc]
		x_nc = x[:,:, self.eff_hist_c:]
		x_nc = torch.flip(x_nc, (2,))

		x_c = self.tcn_causal(x_c)
		x_nc = self.tcn_noncausal(x_nc)

		if any(sequence_lens):
			y1_c = torch.cat([x_c[i, :, self.eff_hist_c:self.eff_hist_c+sequence_lens[i]].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:self.eff_hist_nc+sequence_lens[i]].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()    
		else:
			y1_c = torch.cat([x_c[i, :, self.eff_hist_c:].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()

		y1 = torch.cat((y1_c, y1_nc), dim = 1).contiguous()

		return self.linear(y1)

class TransformerModel_2(nn.Module):
	def __init__(self, input_size_c, input_size_nc, output_size, num_channels_c, num_channels_nc, ksize_c, ksize_nc, dropout, eff_hist_c, eff_hist_nc, d_model=128, nhead=2, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512):
		super(TransformerModel_2, self).__init__()
		
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		self.input_linear_c = nn.Linear(input_size_c, d_model)
		self.input_linear_nc = nn.Linear(input_size_nc, d_model)
		
		encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
		decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
		
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
		
		self.output_linear = nn.Linear(d_model * 2, output_size)  # *2 because we concat causal and non-causal
		
		self.d_model = d_model
		self.eff_hist_c = eff_hist_c
		self.eff_hist_nc = eff_hist_nc
		self.init_weights()

	def init_weights(self):
		initrange = 0.01
		self.input_linear_c.weight.data.normal_(0, initrange)
		self.input_linear_nc.weight.data.normal_(0, initrange)
		self.output_linear.weight.data.normal_(0, initrange)

	def forward(self, x, sequence_lens=[]):
		# Split into causal and non-causal parts
		x_c = x[:, :, :-self.eff_hist_nc]
		x_nc = x[:, :, self.eff_hist_c:]
		x_nc = torch.flip(x_nc, (2,))

		# Reshape and project inputs
		x_c = x_c.permute(2, 0, 1)  # (seq_len, batch, features)
		x_nc = x_nc.permute(2, 0, 1)
		
		x_c = self.input_linear_c(x_c)
		x_nc = self.input_linear_nc(x_nc)
		
		# Add positional encoding
		x_c = x_c * math.sqrt(self.d_model)
		x_nc = x_nc * math.sqrt(self.d_model)
		x_c = self.pos_encoder(x_c)
		x_nc = self.pos_encoder(x_nc)

		# Transform both streams
		memory_c = self.transformer_encoder(x_c)
		memory_nc = self.transformer_encoder(x_nc)

		# Handle sequence lengths
		if any(sequence_lens):
			y1_c = torch.cat([memory_c[self.eff_hist_c:self.eff_hist_c+sequence_lens[i], i] for i in range(memory_c.size(1))])
			y1_nc = torch.cat([memory_nc[self.eff_hist_nc:self.eff_hist_nc+sequence_lens[i], i] for i in range(memory_nc.size(1))])
		else:
			y1_c = torch.cat([memory_c[self.eff_hist_c:, i] for i in range(memory_c.size(1))])
			y1_nc = torch.cat([memory_nc[self.eff_hist_nc:, i] for i in range(memory_nc.size(1))])

		# Concatenate causal and non-causal features
		y1 = torch.cat((y1_c, y1_nc), dim=1).contiguous()
		
		# Final projection to output size
		return self.output_linear(y1)
	
class TransformerModel(nn.Module):
	def __init__(self, input_size_c, output_size, eff_hist_c, eff_hist_nc, d_model=512, nhead=8, num_encoder_layers=3, 
				 num_decoder_layers=3, dim_feedforward=512, dropout=0.1, **kwargs):
		super(TransformerModel, self).__init__()
		
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		self.input_linear = nn.Linear(input_size_c, d_model)
		
		encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
		decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
		
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
		
		self.output_linear = nn.Linear(d_model, output_size)
		
		self.d_model = d_model
		self.eff_hist_c = eff_hist_c
		print(eff_hist_c)
		self.eff_hist_nc = eff_hist_nc
		print(eff_hist_nc)
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.input_linear.weight.data.uniform_(-initrange, initrange)
		self.output_linear.weight.data.uniform_(-initrange, initrange)

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def forward(self, src, sequence_lens=[]):
		# src shape: (batch, features, seq_len) -> (seq_len, batch, features)
		src = src.permute(2, 0, 1)
		
		# Linear projection to d_model dimensions 
		src = self.input_linear(src)
		
		# Add positional encoding
		src = src * math.sqrt(self.d_model)
		src = self.pos_encoder(src)
		
		# Generate target sequence for decoder (initially zeros)
		tgt = torch.zeros_like(src)
		tgt = self.pos_encoder(tgt)
		
		# Generate mask for decoder
		tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
		
		# Encode and decode
		memory = self.transformer_encoder(src)
		output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
		
		# Linear projection to output size
		output = self.output_linear(output)
		
		# Handle sequence lengths and effective history if provided
		if any(sequence_lens):
			output = torch.cat([output[self.eff_hist_c:self.eff_hist_c+sequence_lens[i], i, :] for i in range(len(sequence_lens))])
		else:
			output = output[self.eff_hist_c+self.eff_hist_nc:].permute(1, 0, 2).contiguous()
			output = output.view(-1, output.size(2))
			
		return output

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=10000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)

'''
		y1_c = x_c[-1,-1,self.eff_hist_c:]
		y1_nc = x_nc[-1,-1,-(self.eff_hist_nc+1):]

		y1 = torch.cat([y1_c, y1_nc], 0)
		print(y1.shape)

		print(self.num_channels_c, self.num_channels_nc)

		start_time = time.time()
'''


'''
		if any(sequence_lens):
			#y1_c = torch.cat([y1_c[self.eff_hist_c:self.eff_hist_c+sequence_lens[i]].contiguous() for i in range(y1_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			#y1_nc = torch.cat([y1_nc[self.eff_hist_nc:self.eff_hist_nc+sequence_lens[i]].contiguous() for i in range(y1_nc.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_c = torch.cat([y1_c[i, :, self.eff_hist_c:self.eff_hist_c+sequence_lens[i]].contiguous() for i in range(y1_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_nc = torch.cat([y1_nc[i, :, self.eff_hist_nc:self.eff_hist_nc+sequence_lens[i]].contiguous() for i in range(y1_nc.shape[0])], dim=1).transpose(0, 1).contiguous()

		else:
			#y1_c = torch.cat([y1_c[self.eff_hist_c:].contiguous() for i in range(y1_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			#y1_nc = torch.cat([y1_nc[self.eff_hist_nc:].contiguous() for i in range(y1_nc.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_c = torch.cat([y1_c[i, :, self.eff_hist_c:].contiguous() for i in range(y1_c.shape[0])], dim=1).transpose(0, 1).contiguous()
			y1_nc = torch.cat([y1_nc[i, :, self.eff_hist_nc:].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()
'''




class LossFunctions():

	class MSELoss():
		def __init__(self, weight_importance=False):
			self.weight_importance = weight_importance

		def forward(self, y1, y2, y_var=[]):
			# if y_var.shape == y1.shape:
			# 	mse = torch.mean(((y1-y2)**2)/y_var)
			# else:
			# 	mse = torch.mean((y1-y2)**2)
			err = torch.abs(y1-y2)
			# if y_var.shape == y1.shape:
			if self.weight_importance:
				err /= y_var
			mse = torch.mean(err**2)
			return mse

	class SmoothL1Loss():
		def __init__(self, weight_importance=False):
			self.weight_importance = weight_importance

		def forward(self, y1, y2, y_var):
			err = torch.abs(y1-y2)

			# if y_var.shape == y1.shape:
			if self.weight_importance:
				err /= y_var

			l1 = err-0.5
			l2 = 0.5*(err**2)
			mask = err < 1

			huber = l1 
			huber[mask] = l2[mask]
			huber = torch.mean(huber)
			return huber
