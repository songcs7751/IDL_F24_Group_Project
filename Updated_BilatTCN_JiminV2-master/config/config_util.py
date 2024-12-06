from dataclasses import dataclass
import argparse
import importlib

# config_util.py
import os

@dataclass
class Config():
	run_name = 'baseline'
	input_dir = os.path.abspath(r"/root/IDL_F24_Group_Project/Updated_BilatTCN_JiminV2-master/doi_10_5061_dryad_8kprr4xsv__v20240321")
	output_dir = os.path.abspath(r"/root/IDL_F24_Group_Project/Updated_BilatTCN_JiminV2-master/output")
    # 나머지 설정은 그대로 유지
#class Config():
	# Directories and Write Flags
#	input_dir = r"Users/wlals/Downloads/ProcessedData" # 키튼 데이터셋 인풋 데이터 디렉토리 C:\Users\wlals\Downloads\ProcessedData
##	output_dir = r"Users/wlals/Downloads" # Estimated hip moment output 파일 디렉토리
    
	write_output = True
	save_model = False

	# Gait Modes and Trial Types
	gait_modes = ['LG']
	# trials_use = ['_l_', '_r_'] # 이것도 우리한테 필요없고 
	# trials_use = ['normal_walk_1_0-6', 'normal_walk_1_1-2', 'normal_walk_1_1-8']
	# trials_ignore = ['_ZI_', '_NE_'] # 필요없고
	trials_use = ['exo']
	trials_ignore = []
	# label = 'hip_flexion_moment'
	label = 'hip_flexion_l_moment'

	# Subjects
	# subjects = ['AB01', 'AB02', 'AB03', 'AB05', 'AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13']
	subjects = ['AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20'] 
	# subjects = ['AB11', 'AB12']

	# Input sensors
	# sensors = ['hip_sagittal', 'd_hip_sagittal_lpf', 'thigh_accel', 'thigh_gyro', 'pelvis_accel', 'pelvis_gyro']
	# sensors = ['Pelvis_V_ACCX', 'Pelvis_V_ACCY', 'Pelvis_V_ACCZ', 'Pelvis_V_GYROX', 'Pelvis_V_GYROY', 'Pelvis_V_GYROZ', 
	# 		'LThigh_V_ACCX', 'LThigh_V_ACCY', 'LThigh_V_ACCZ', 'LThigh_V_GYROX', 'LThigh_V_GYROY', 'LThigh_V_GYROZ',
	# 		'RThigh_V_ACCX', 'RThigh_V_ACCY', 'RThigh_V_ACCZ', 'RThigh_V_GYROX', 'RThigh_V_GYROY', 'RThigh_V_GYROZ',
	# 		'hip_flexion_r', 'hip_flexion_l', 'hip_flexion_l_moment']
	# sensors = ['Pelvis_V_ACCX', 'Pelvis_V_ACCY', 'Pelvis_V_ACCZ', 'Pelvis_V_GYROX', 'Pelvis_V_GYROY', 'Pelvis_V_GYROZ', 
	# 		'LThigh_V_ACCX', 'LThigh_V_ACCY', 'LThigh_V_ACCZ', 'LThigh_V_GYROX', 'LThigh_V_GYROY', 'LThigh_V_GYROZ',
	# 		'RThigh_V_ACCX', 'RThigh_V_ACCY', 'RThigh_V_ACCZ', 'RThigh_V_GYROX', 'RThigh_V_GYROY', 'RThigh_V_GYROZ',
	# 		'hip_flexion_r', 'hip_flexion_l']
	sensors = ['enc_angle_l','enc_angle_r','enc_velo_l','enc_velo_r',
            'thigh_accel_x_l','thigh_accel_y_l','thigh_accel_z_l','thigh_gyro_x_l','thigh_gyro_y_l','thigh_gyro_z_l',
            'thigh_accel_x_r','thigh_accel_y_r','thigh_accel_z_r','thigh_gyro_x_r','thigh_gyro_y_r','thigh_gyro_z_r',
            'pelvis_accel_x','pelvis_accel_y','pelvis_accel_z','pelvis_gyro_x','pelvis_gyro_y','pelvis_gyro_z']
	sensors_ignore = ['_raw', '_filt', '_orig']
	# 이 부분 내가 지우고 알아서 데이터 column 이름으로 만들면 될듯

	# Network Training/Testing Parameters
	# augmentation = False
	# standardize = False
	# weight_decay = 0 # This is only used if opt = 'AdamW', 0.1
	num_epochs = 50
	min_epochs = 10
	steps_per_batch = 512
	batch_size_per_step = 512
	early_stopping = True
	patience = 5 # Only used if early_stopping = True


	# Network Hyperparameters
	ksize = [4]
	hsize = [50]
	loss = ['MSELoss'] # Current options are 'MSELoss' and 'SmoothL1Loss'
	dropout  = [0.3]
	lr = [0.0005]
	pred = [0]
	eff_hist_limit = 400 # This is based on the padding set when extracting step data

	# Original Settings
	# levels = [5]
	# opt = ['Adam']
	# eff_hist_limit = 200 # This is based on the padding set when extracting step data
	# steps_per_batch = 256
	# batch_size_per_step = 32
	# patience = 50 # Only used if early_stopping = True
	# batch_pad_value = -500

	# #Improved Version
	augmentation = True
	standardize = True
	weight_decay = 0.1 # This is only used if opt = 'AdamW', 0.1
	steps_per_batch = 512
	batch_size_per_step = 512
	patience = 5 # Only used if early_stopping = True
	batch_pad_value = -5000
	levels = [6]
	opt = ['AdamW']

	# Run Details
	enable_cuda = True
	max_device_jobs = 1
	
config = Config()

def load_config(config_file_name):
	if config_file_name.endswith('.py'): config_file_name = config_file_name[:-3]

	print(f"Using {config_file_name}.")
	module = importlib.import_module('.' + config_file_name, package = 'config')
	return module.config


def parse_args():
	parser = argparse.ArgumentParser()

	# Handle CLI arguments
	parser.add_argument('--config', type = str, default = 'default_config', dest = 'config_file_name')

	return parser.parse_args()


def load_config_file():
#	args = parse_args()
#	config = load_config(**vars(args))
	config = Config()	
	m_dict = {	
				'run_name': config.run_name,
				'augmentation': config.augmentation,
				'standardize': config.standardize,
				'weight_decay': config.weight_decay,
				'input_dir': config.input_dir,
				'output_dir': config.output_dir,
				'write_output': config.write_output,
				'save_model': config.save_model,
				'gait_modes': config.gait_modes,
				'trials_use': config.trials_use,
				'trials_ignore': config.trials_ignore,
				'label': config.label,
				'subjects': config.subjects,
				'sensors': config.sensors,
				'sensors_ignore': config.sensors_ignore,
				'num_epochs': config.num_epochs,
				'min_epochs': config.min_epochs,
				'steps_per_batch': config.steps_per_batch,
				'batch_size_per_step': config.batch_size_per_step,
				'early_stopping': config.early_stopping,
				'patience': config.patience,
				'batch_pad_value': config.batch_pad_value,
				'eff_hist_limit': config.eff_hist_limit
			}

	h_dict = {
				#'ksize_c': config.ksize_c,
				#'ksize_nc': config.ksize_nc,
				'ksize_c': config.ksize,
				'ksize_nc': config.ksize,
				'hsize': config.hsize,
				#'levels_c': config.levels_c,
				#'levels_nc': config.levels_nc,
				'levels_c': config.levels,
				'levels_nc': config.levels,
				'loss': config.loss,
				'opt': config.opt,
				'dropout': config.dropout,
				'lr': config.lr,
				'pred': config.pred
			}

	r_dict = {
				'enable_cuda': config.enable_cuda,
				'max_device_jobs': config.max_device_jobs
			}

	return r_dict, m_dict, h_dict