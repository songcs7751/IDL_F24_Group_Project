import numpy as np
import pandas as pd
from biomechdata import AbleBodyDataset2, BilatTCN, LossFunctions, get_model_dict
from biomechutils import DeviceManager, get_file_names, update_rel_dir, build_exp, get_exp_name, write_to_file
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from os import path, listdir, makedirs
import argparse
import time
# import multiprocessing as mp
import torch.multiprocessing as mp
import signal
import traceback
from copy import deepcopy
from config.config_util import load_config_file
import csv
import os

def init_worker():
	# Ignore normal keyboard interrupt exit to properly close multiprocessing workers 
	# 멀티 프로세싱 환경에서 각 워커 프로세스 초기화 및 키보드 인터럽트 무시 코드
	signal.signal(signal.SIGINT, signal.SIG_IGN)


def log_result(result):
	# 각 워커 프로세스 결과 수집후 global variable result에 속속 추 98.6
	global results 
	results.append(result)
	
def pad_data(data, des_length, pad_value, dim=0):
	# 주어진 데이터를 특정 길이로 패딩
	# data = 입력 데이터, des_length = 패딩 후 목표 데이터 길이, pad_value = 패딩 값, dim = 패딩할 차원
	start_time = time.time()
	device = data.device # 입력 데이터의 디바이스(CPU 또는 GPU) 추출
	if dim == 0: # 만약 패딩할 차원이 0 이라면
		# data_padded = torch.cat((data, torch.ones_like(data)*pad_value), dim=0)[:des_length, :].contiguous()
		data_padded = (torch.ones((des_length, data.shape[1]))*pad_value).to(device)
		# 패딩된 데이터를 저장할 텐서 data_padded = (des_length, data.shape[1])의 크기를 가지며 pad_value로 채원짐
		data_padded[0:data.shape[0], :] = data
		# 기존 데이터를 패딩된 텐서의 시작 부분에 복사
	elif dim == 1:
		# data_padded = torch.cat((data, torch.ones_like(data)*pad_value), dim=1)[:, :des_length].contiguous()
		data_padded = (torch.ones((data.shape[0], des_length))*pad_value).to(device)
		# (data.shape[0], des_length)의 크기를 가지며 pad_value로 채워짐
		data_padded[:, 0:data.shape[1]] = data
	else:
		print('Cannot pad in this dimension.')
		return data 
	return data_padded, time.time()-start_time # 패딩된 데이터와 패딩에 소요된 시간 반환


def train_test_subject_ind(input_dir, test_subject, train_subjects, gait_modes, sensors, sensors_ignore, device_name, 
	batch_size_per_step, steps_per_batch, num_epochs, min_epochs, batch_pad_value, early_stopping, patience, 
	model_dict, label, loss, opt, lr, pred, eff_hist_limit, trials_use, trials_ignore, write_output, save_model, 
	output_dir='', output_name=''):
		
	try:
		if not any(device_name):
			print('No device name.')
			return -1
		if not any(test_subject):
			print('No test subject.')
			return -1
		if not any(train_subjects):
			print('No train subjects.')
			return -1
		if not any(label):
			print('No label.')
			return -1
		# "Check device, test subject, training subjects, and label. If any of them are empty, return -1 and exit the function."

		if model_dict['eff_hist_c'] > eff_hist_limit:
			print('Effective causal history too long.')
		if model_dict['eff_hist_nc'] > eff_hist_limit:
			print('Effective noncausal history too long.')
			return -1
		eff_hist = model_dict['eff_hist_c'] + model_dict['eff_hist_nc']
		# Check the history length of causal and non-causal?
		# If it exceeds the specified limit eff_hist_limit, print an error message and exit the function, then calculate the eff_hist length

		device = torch.device(device_name)
		cpu = torch.device('cpu')
		# Set up the device. Determine whether to use GPU or CPU based on device_name.

		# Limit PyTorch to 1 thread per process (keeps each model isolated on a thread, which improves parallel computing benefits)
		torch.set_num_threads(1)
		# Set the number of threads to use in PyTorch to 1

		# Set up train dataset	
		# sub_dir = input_dir.split('*')[-1][1:]
		# input_dir = input_dir.split('*')[0][:-1]

		# Use input_dir as is without changing sub_dir
		sub_dir = ''
		# train_dataset = AbleBodyDataset2(input_dir, label=label, ft_use=sensors, ft_ignore=sensors_ignore, trials_use=trials_use, trials_ignore=trials_ignore)
		train_dataset = AbleBodyDataset2(input_dir=input_dir, label=label, ft_use=sensors, ft_ignore=sensors_ignore, trials_use=trials_use, trials_ignore=trials_ignore)
		# Set up the training dataset using AbleBodyDataset2 from biomechdata
		# This line determines which dataset I will use
		print(f"Train dataset setup: input_dir={input_dir}, label={label}, sensors={sensors}")
 
		# Get all trial names for train dataset
		for s in train_subjects:
			print(f'Train subject now is: {s}')
			# train_dataset.add_trials(sub_dir=s + sub_dir, ext='.csv', include=gait_modes)
			train_dataset.add_trials(sub_dir=s + sub_dir, ext='.csv')

		# Add experimental data in CSV format containing the specified gait modes for the training subjects to the training dataset
		print(f"Added trials for subjects: {train_subjects}")
		print(f"Total trials in train dataset: {len(train_dataset.trial_names)}")

		# Load train dataset into memory
		print(f'{test_subject} loading train set.')
		train_dict, train_trial_list = train_dataset.load_trial_dict(eff_hist=eff_hist, eff_pred=pred,
			batch_size_per_trial=batch_size_per_step, device=device, verbose=False)
		# Load training dataset: Based on the specified history and prediction range,
		# load the training dataset into memory, and return a dictionary (train_dict) containing data for each experiment and a list of experiments (train_trial_list)
		print(f"Loaded train data: Number of trials loaded = {len(train_dict)}")
		if train_trial_list:
			print(f"First trial's data shape: {train_dict[train_trial_list[0]]['data'].shape}")

		# Randomize the training trial list
		np.random.shuffle(train_trial_list) # Randomly shuffle the training trials to eliminate the effect of order

		# Set up test dataset and get all trial names
		# test_dataset = AbleBodyDataset2(input_dir, label=label, ft_use=sensors, ft_ignore=sensors_ignore, trials_use=trials_use, trials_ignore=trials_ignore)
		test_dataset = AbleBodyDataset2(input_dir=input_dir, label=label, ft_use=sensors, ft_ignore=sensors_ignore, trials_use=trials_use, trials_ignore=trials_ignore)

		# test_dataset.add_trials(sub_dir=os.path.join(test_subject, sub_dir), ext='.csv', include=gait_modes)
		test_dataset.add_trials(sub_dir=os.path.join(test_subject, sub_dir), ext='.csv')
		# Set up the test dataset and add experiments: Create a dataset for the test data,
		# and add experimental data containing the specified gait modes.
		print(f"Added trials for test subject: {test_subject}")
		print(f"Total trials in test dataset: {len(test_dataset.trial_names)}")


		# Load test dataset into memory
		print(f'{test_subject} loading test set. from {test_dataset.input_dir}')
		x_test, y_test, y_test_step_count = test_dataset.load_trial_list(eff_hist=eff_hist, eff_pred=pred,
			device=device, verbose=False)
		
		# Everything below here is written by me
		# print(f"x_test: {x_test}")
		# print(f"y_test: {y_test}")
		# print(f"y_test_step_count: {y_test_step_count}")
		
		print(f"Loaded test data: Number of test trials = {len(x_test)}")
		if x_test:
			print(f"First test trial's data shape: {x_test[0].shape}, labels shape: {y_test[0].shape}")
		else:
			print(f"No test data loaded for {test_subject}.")

        # Step 6: Reshape y_test and y_test_step_count for training and logging data
		# if y_test:
		# 	y_test = torch.cat(tuple(y_test), dim=0) # 여기서 튜플로 바꾸었음
		# 	y_test_step_count = np.asarray(y_test_step_count).reshape(-1, 1)
		# 	print(f"y_test shape after concatenation: {y_test.shape}")
		# 	print(f"y_test_step_count shape: {y_test_step_count.shape}")
		# else:
		# 	print(f"Warning: y_test is empty!")
		# END OF MY CODE

		# Reshape y_test and y_test_step_count for training and logging data
			# Keep x_test separated in list since each trial needs to be passed through the TCN one at a time
		
		# y_test = torch.cat(tuple(y_test), dim=0) # 여기서 튜플로 바꾸었음
		y_test_step_count = np.asarray(y_test_step_count).reshape(-1, 1)

		# Set up neural network, optimizer, and loss function
		input_size_c = 22 #train_dict[list(train_dict.keys())[0]]['data'].shape[1] # Get input size from example training trial
		input_size_nc = 22
		input_size = 22
		# 이거를 우리가 원하는 대로 바꿔야 할텐데
		output_size = 1 # Output is joint torque estimation
		model_dict.update({'input_size_c': input_size_c, 'input_size_nc' : input_size_nc, 'output_size': output_size})
		net = BilatTCN(**model_dict).to(device)
		optimizer = getattr(optim, opt)(net.parameters(), lr=lr)
		loss_function = nn.MSELoss()

		# Train Model
		train_loss_arr = [] # Used to construct learning curve
		val_loss_arr = [] # Userd to construct learning curve and for early stopping
		val_loss_r_arr = [] # Used to construct learning curve and for early stopping based on inference from data with randomized environment
		patience_count = 0 # Counter for early stopping criteria
		best_val_loss = float('inf') # Track best validation loss for early stopping criteria
		best_train_loss = float('inf')
		best_y_out = []
		training_results = {}
  
		# x_data = x_test.copy()
		# x_test = []
		# for xd in x_data:
		# 	x_test.append(xd.view(1, xd.shape[0], -1))
		# x_test = torch.cat(x_test, dim=0)
		# print("x_test shape:", x_test.shape)
		# x_test = x_test.transpose(0, 1).view(1, input_size, -1).contiguous()

		# Iterate through all epochs unless early stopping criteria is met
		for epoch in range(1, num_epochs+1):
			print(f"{test_subject} Epoch {epoch}")

			# Start epoch timer
			start_time = time.time() # Used to print epoch time

			# Randomize the training trial list
			np.random.shuffle(train_trial_list)
			print(train_trial_list[0])

			x_train = [] # Container for mini-batch input data
			y_train = [] # Container for mini-batch label data

			step_count = 0 # Counter for keeping track of how many sections of step data have been added to mini-batch
			batch_count = 0 # Count number of mini-batches per epoch to average train mse over epoch for learning curve
			train_loss = 0.0 # Variable to compute the train loss per epoch
			train_mse = 0.0 # Variable to compute the train mse per epoch

			net.train(True) # This enables gradient tracking and training parameters (e.g. dropout)
			sequence_lens = []
			

			# Iterate through the randomized trial list to build mini-batch
			for i, trial in enumerate(train_trial_list):
				step_count += 1 # Increment counter for keeping track of steps per mini-batch

				# Decide which slice to section out from the next step (this lets mini-batches have more varied step data without having to be too large)
				slice_count = train_dict[trial]['slice_count']
				slice_order = train_dict[trial]['slice_order'][slice_count]
				train_dict[trial]['slice_count'] += 1
				if train_dict[trial]['slice_count'] >= len(train_dict[trial]['slice_order']):
					train_dict[trial]['slice_count'] = 0

				# Find start and end of the input data and label for this slice
				y_start = slice_order * batch_size_per_step
				y_end = (slice_order + 1) * batch_size_per_step
				x_start = y_start
				x_end = y_end + eff_hist

				# Get slice data and transpose input data for conv
				x_data = train_dict[trial]['data'][x_start:x_end, :].transpose(0, 1)
				y_data = train_dict[trial]['labels'][y_start:y_end]

				# print('y_data', y_data.shape)
				sequence_lens.append(y_data.shape[0])

				# If this slice is the end of a step, it may need padding to maintain the same shape as the other slices in the mini-batch
				if y_data.shape[0] < batch_size_per_step:
					x_data, _ = pad_data(x_data, batch_size_per_step+eff_hist, batch_pad_value, dim=1)
					# pad_data(data, des_length, pad_value, dim=0):
				# Append each slice to mini-batch list to be concatenated later
				x_train.append(x_data.view(1, x_data.shape[0], -1)) # Correct shape for conv
				y_train.append(y_data)
				

				# If mini-batch is full or we have passed through all of the training data, train on mini-batch
				if step_count >= steps_per_batch or i >= len(train_trial_list) - 1:
					# Concatenate training data into required tensors
					x_train = torch.cat(x_train, dim=0) # shape = (num_steps, num_features, num_points)
					y_train = torch.cat(y_train, dim=0) # shape = (num_steps*num_points, 1)
					# if len(train_trial_list) - i  < 10:
					# 	print(x_train.shape, x_train)
					# Clear gradients store in optimizer and complete forward pass
					optimizer.zero_grad()
					y_out = net(x_train, sequence_lens)
					#
					# y_out torch.Size([2003, 1])
					# x_train torch.Size([64, 22, 404])
					# y_train torch.Size([2005, 1])
					# print("x_train", x_train.shape)
					# print("y_train", y_train.shape)
					# print("sequence_lens", len(sequence_lens))
					# print("y_out", y_out.shape)
     
					# print("x_train", np.isnan(x_train.cpu().detach().numpy()).any())
					# print("sequence_lens", np.isnan(np.ndarray(sequence_lens)).any(), sequence_lens[:10], np.ndarray(sequence_lens))
					# print("y_out", np.isnan(y_out.cpu().detach().numpy()).any())

					# Backpropagation
					loss = loss_function.forward(y_out, y_train) # Compute loss
					loss.backward() # Compute gradient
					optimizer.step() # Update model weights

					# Track training loss
					train_loss += loss.item()
					batch_count += 1

					# Reset training data containers/variables for next mini-batch
					x_train = []
					y_train = []
					step_count = 0
					sequence_lens = []

			# Compute the training loss after each epoch
			train_loss /= batch_count
			train_loss_arr.append(train_loss)

			# Compute the validation loss after each epoch
			net.train(False) # Disable gradient tracking and training parameter
     
			
			
			sequence_lens_test = []
			for i in range(len(y_test_step_count)):
				sequence_lens_test.append(y_test_step_count[i][0])
			sequence_lens_test.append(y_data.shape[0])
			# print("x_test shape:", x_test.shape)
			# print("y_test shape:", y_test.shape)
			# print("sequence_lens_test:", len(sequence_lens_test))
			val_loss = 0
			with torch.no_grad():
				for i,x_test_trial in enumerate(x_test):
					x_test_trial = x_test_trial.unsqueeze(0).transpose(1, 2).contiguous()
					# print("x_test_trial shape:", x_test_trial.shape)
					
					# print("y_test shape:", y_test[i].shape)
						
					y_out = net(x_test_trial)
					# print("y_out shape:", y_out.shape)
					loss = loss_function.forward(y_out, y_test[i])
					# print("val_loss:", val_loss, loss.item())
					val_loss += loss.item()
     
				val_loss /= len(x_test)
					
     
				# y_out = net(x_test, sequence_lens_test)
				# print("y_out shape:", y_out.shape)
				# val_loss = loss_function.forward(y_out, y_test)
				val_loss_arr.append(val_loss)

			# Track best validation loss for early stopping
			if early_stopping and val_loss < best_val_loss:
				best_val_loss = val_loss
				best_train_loss = train_loss
				patience_count = 0
				best_y_out = y_out # Saving best validation estimate
				if save_model:
					model_dict['state_dict'] = deepcopy(net.state_dict())
			else:
				patience_count += 1

			print(f"{test_subject} Epoch {epoch}: Train Loss {round(train_loss, 4)}, Val Loss {round(val_loss, 4)}, Patience {patience_count}/{patience}, {round(time.time()-start_time, 1)} seconds")
			
			# Save validation loss for the current epoch
			csv_file_path = "training_result.csv"

			# Check if the file exists to determine if we need to write the header
			file_exists = os.path.isfile(csv_file_path)

			with open(csv_file_path, mode='a', newline='') as csv_file:
				writer = csv.writer(csv_file)
				
				# Write the header row if the file does not exist
				if not file_exists:
					header = ['Subject', 'Epoch', 'Train loss', 'Val Loss']
					writer.writerow(header)
				
				# Write the current epoch's validation loss
				writer.writerow([test_subject, epoch, train_loss, val_loss])

			print(f"Validation loss for epoch {epoch} has been saved to {csv_file_path}")
   
			# Stop training if early stopping criteria is met
			if early_stopping and patience_count >= patience and epoch >= min_epochs:
				print('Early Stopping!')
				break

		if not early_stopping:
			best_val_loss = val_loss
			best_train_loss = train_loss
			best_y_out = y_out # Saving best validation estimate
			if save_model:
				model_dict['state_dict'] = deepcopy(net.state_dict())

		if False:
			print(f"Saving {output_name} to {output_dir}.")
			if not path.exists(output_dir):
				makedirs(output_dir)
			summary_file_name = output_dir + '/' + output_name + '.csv'
			summary_header = ['subject', 'val_loss', 'train_loss', 'train_subjects'] + ['ft_use', 'ft_ignore', 
				'batch_size_per_step', 'steps_per_batch', 'num_epochs', 'pred', 'eff_hist_c', 'eff_hist_nc', 'kernel_size', 'hidden_size',               
				'optimizer', 'lr', 'loss', 'levels']
			summary_data = [test_subject, str(best_val_loss.item()), str(best_train_loss), ('-').join(train_subjects)]
			summary_data += [('-').join(train_dataset.features_use), ('-').join(sensors_ignore), batch_size_per_step, 
				steps_per_batch, epoch, pred, eff_hist, model_dict['ksize_c'], model_dict['ksize_nc'], model_dict['num_channels_c'][0], model_dict['num_channels_nc'][0], 
				opt, lr, loss.item(), len(model_dict['num_channels_c']), len(model_dict['num_channels_nc'])]
			write_to_file(summary_file_name, summary_header, write_type='w', print_msg=False)
			write_to_file(summary_file_name, summary_data, write_type='a', print_msg=False)

			ts_dir = output_dir + '/TimeSeries'
			ts_file_name = ts_dir + '/' + output_name + '.csv'
			if not path.exists(ts_dir):
				makedirs(ts_dir)
			ts_trials = ['Order: '] + test_dataset.trial_names
			write_to_file(ts_file_name, ts_trials, write_type='w', print_msg=False)
			
			# y_test, best_y_out, y_test_step_count가 2차원 배열인지 확인
			if y_test.dim() == 1:
				print("y_test dimension is 1!")
				y_test = y_test[:, np.newaxis]
			if best_y_out.dim() == 1:
				print("best_y_out dimension is 1!")
				best_y_out = best_y_out[:, np.newaxis]
			if len(y_test_step_count.shape) == 1:
				print("y_test_step_count.shape length is 1!")
				y_test_step_count = y_test_step_count[:, np.newaxis]

			test_out = np.concatenate((y_test.cpu().numpy(), best_y_out.cpu().numpy(), y_test_step_count), axis=1)

			test_out = np.concatenate((y_test.to(cpu).numpy(), best_y_out.to(cpu).numpy(), y_test_step_count), axis=1)
			ts_df = pd.DataFrame(test_out, columns=['label', 'estimate', 'step_count'])
			ts_df.to_csv(ts_file_name, mode='a', index=False)

			lc_dir = output_dir + '/LearningCurve'
			lc_file_name = lc_dir + '/' + output_name + '.csv'
			if not path.exists(lc_dir):
				makedirs(lc_dir)
			lc_df = pd.DataFrame(np.concatenate((np.asarray(train_loss_arr).reshape(-1, 1), np.asarray(val_loss_arr).reshape(-1, 1)), axis=1), columns=['train_loss', 'val_loss'])
			lc_df.to_csv(lc_file_name, index=False)

		if save_model:
			model_dir = output_dir + '/SavedModels'
			model_file_name = model_dir + '/' + output_name + '.tar'
			if not path.exists(model_dir):
				makedirs(model_dir)
			torch.save(model_dict, model_file_name)

		if 'cuda' in device_name:
			return int(device_name[-1])
		else:
			return 0

	except:
		print(f"{test_subject} Error!")
		traceback.print_exc()
		return -1

def main(m_dict, h_dict, enable_cuda = False, max_device_jobs = 1):
	mp.set_start_method('spawn')

	# Global variable used to store results from parallel processes
	global results

	subjects = m_dict['subjects']
	for s in subjects:
		print(f'WE HAVE SUBJECTS: {s}')

	del m_dict['subjects']

	# Update and create directories
	m_dict['input_dir'] = update_rel_dir(m_dict['input_dir'])
	m_dict['output_dir'] = update_rel_dir(m_dict['output_dir'], mkdir=True)

	# Set up device manager for distributing runs
	dm = DeviceManager(use_cuda = enable_cuda, max_device_jobs = max_device_jobs)

	# Set up experiments for each set of hyperparameters
	exp_list = build_exp(h_dict)
 
	# Initialize workers for multiprocessing
	pool = mp.Pool(8, initializer=init_worker, maxtasksperchild=1) # Change the number of workers here! (32 for ML CPU)

	# Wrap everything in try-except to handle keyboard interrupt
	try:
		for exp_conds in exp_list:
			exp_name = get_exp_name(exp_conds)

			exp_conds['model_dict'] = get_model_dict('BilatTCN', exp_conds)
			del exp_conds['dropout']
			del exp_conds['hsize']
			del exp_conds['ksize_c']
			del exp_conds['ksize_nc']
			del exp_conds['levels_c']
			del exp_conds['levels_nc']

			for s in subjects:
				run_name = s + '_' + exp_name
				if run_name + '.csv' in listdir(m_dict['output_dir']):
					print('Skipping ' + run_name)
					continue

				while not dm.device_available():
#					time.sleep(2)
					paused_results = results.copy()
					err = dm.update_devices(paused_results)
					if not err:
						print('Return error!')
						raise
				m_dict['device_name'] = dm.get_next_device()

#				time.sleep(1)

				train_subjects = deepcopy(subjects)
				train_subjects.remove(s) # 이게 subject에서 첫번째 서브젝을 트레이닝 데이터에서 remove시켜버림

				run_dict = deepcopy(m_dict)
				run_dict.update(exp_conds)
				run_dict.update({'output_name': run_name})
				run_dict.update({'test_subject': s})
				run_dict.update({'train_subjects': train_subjects})

				print("config", run_dict)

				pool.apply_async(train_test_subject_ind, kwds=deepcopy(run_dict), callback=log_result)

		while dm.get_active_process_count() > 0:
#			time.sleep(2)
			paused_results = results.copy()
			err = dm.update_devices(paused_results)
			if not err:
				print('Return error!')
				raise

	except Exception as e:
		traceback.print_exc()
		pool.terminate()
		pool.join()
		return

	return


if __name__=="__main__":
	# Global variable used to store results from parallel processes
	global results
	results = []

	r_dict, m_dict, h_dict = load_config_file()
	main(m_dict, h_dict, **r_dict)