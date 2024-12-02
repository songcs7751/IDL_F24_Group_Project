from os import path, listdir, makedirs, getcwd
import torch
import itertools
import numpy as np
from scipy import signal
import collections


def build_exp(exp_dict):
	keys, values = zip(*exp_dict.items())
	return [dict(zip(keys, v)) for v in itertools.product(*values)]

def get_exp_name(exp_dict, rename=[]):
	keys = np.sort(list(exp_dict.keys()))
	if any(rename):
		keys = [k for k in keys if k not in list(dict(rename).keys())]
		exp_name = ('_').join([('').join([r[1], str(exp_dict[r[0]])]) for r in rename])
		exp_name += '_' + ('_').join([('').join([k, str(exp_dict[k])]) for k in keys])
	else:
		exp_name = ('_').join([('').join([k, str(exp_dict[k])]) for k in keys])
	return exp_name

def get_file_names(input_dir, sub_dir='', ext=''):
	print('get_file_names:', input_dir, sub_dir, ext)
	sub_dir = [d for d in sub_dir if d != '.DS_Store']
	# 여기 있는 부호는 /가 맞으니까 건들 ㄴㄴ
	# C:/Users/wlals/Downloads/ProcessedData/AB02/LG/AB02_LG_normal_walk_1_1-8.csv -->  C:/Users/wlals/Downloads/ProcessedData/AB02/LG/AB02_LG_normal_walk_1_1-8.csv
	if not any(sub_dir):
		file_names = listdir(input_dir)
		print('listdir file_names:', file_names)
	else:
		if not isinstance(sub_dir, list):
			sub_dir = [sub_dir]
		sub_dir = [d for d in sub_dir if path.exists(input_dir + '/' + d)]
		file_names = [d + '/' + n for d in sub_dir for n in listdir(input_dir + '/' + d)]
	if any(ext):
		file_names = [n for n in file_names if ext in n]
	return file_names

def norm_train_dict(train_dict, subject_info, name):
	for k in train_dict:
		s = k.split('/')[0]
		train_dict[k]['labels'] /= subject_info.loc[s, name]

def update_rel_dir(d, mkdir=False):
	d = [d if ':' in d or '/' == d[0] else getcwd()+'/'+d][0]
	if mkdir and not path.exists(d):
		makedirs(d)
	return d

def write_to_file(file_name, msg, write_type='a', print_msg=True):
	with open(file_name, write_type) as f:
		if isinstance(msg, list):
			for i,packet in enumerate(msg):
				f.write(str(packet))
				if i < len(msg)-1:
					f.write(',')
		else:
			f.write(str(msg))
		f.write('/n')
	if print_msg:
		print(msg)
	return True

class DeviceManager():
	def __init__(self, use_cuda=True, max_device_jobs=8):
		num_devices = torch.cuda.device_count()
		if num_devices == 0:
			num_devices = 1
		self.device_jobs = [0]*num_devices
		self.num_completed_jobs = 0
		self.max_device_jobs = max_device_jobs
		self.use_cuda = use_cuda

	def device_available(self):
		next_device = np.argmin(self.device_jobs)
		if self.device_jobs[next_device] < self.max_device_jobs:
			return True
		else:
			return False

	def get_active_process_count(self):
		return sum(self.device_jobs)

	def get_next_device(self):
		if self.device_available():
			next_device = np.argmin(self.device_jobs)
			self.device_jobs[next_device] += 1
			if torch.cuda.is_available() and self.use_cuda:
				return 'cuda:' + str(next_device)
			else:
				return 'cpu'
		else:
			return None

	def update_devices(self, results):
		if len(results) > self.num_completed_jobs:
			new_completed_jobs = len(results) - self.num_completed_jobs
			self.num_completed_jobs += new_completed_jobs
			for i in range(1, new_completed_jobs+1):
				if results[-i] >= 0:
					self.device_jobs[results[-i]] -= 1
				else:
					return False
		return True

class Filter(object):
    '''Parent class for filters, to help with type hinting. 
    Note: for filter modularity, all child classes should have a filter() 
    function that takes only the most recent value. This way, different filters 
    can be passed to objects constructors without replacing that class's code'''

    def filter(self, new_val):
        raise ValueError('filter() not implemented for child class of Filter')


class PassThroughFilter(Filter):
    def filter(self, new_val):
        return new_val


class Butterworth(Filter):
    '''Implements a real-time Butterworth filter using second orded cascaded filters.'''

    def __init__(self, N: int, Wn: float, btype='low', fs=None, n_cols=0):
        ''' 
        N: order
        Wn: (default) normalized cutoff freq (cutoff freq / Nyquist freq). If fs is passed, cutoff is in freq.
        btype: 'low', 'high', or 'bandpass'
        fs: Optional: sample freq, Hz. If not None, Wn describes the cutoff freq in Hz
        '''
        self.N = N
        if fs is not None:
            self.Wn = Wn/(fs/2)
        else:
            self.Wn = Wn
        self.btype = btype
        self.sos = signal.butter(N=self.N, Wn=self.Wn,
                                 btype=self.btype, output='sos')
        self.zi = signal.sosfilt_zi(self.sos)
        if n_cols > 0:
            self.zi = np.repeat(self.zi, n_cols, axis=0).transpose().reshape(1, 2, -1)

    def filter(self, new_val, axis=-1):
        filtered_val, self.zi = signal.sosfilt(
            sos=self.sos, x=new_val, zi=self.zi, axis=axis)
        return filtered_val

    def filter_one(self, new_val: float) -> float:
        return self.filter([new_val])[0]


class MovingAverage(Filter):
    '''Implements a real-time moving average filter.'''

    def __init__(self, window_size):
        self.deque = collections.deque([], maxlen=window_size)

    def filter(self, new_val):
        # Optimize for efficiency is window size is large
        self.deque.append(new_val)
        return np.mean(self.deque)

def is_sequence(x):
	return isinstance(x, list) or isinstance(x, np.ndarray)