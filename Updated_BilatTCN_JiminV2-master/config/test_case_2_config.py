from config import config_util


'''
# Directories and Write Flags
input_dir: Directory containing data
output_dir: Results directory
write_output: Write results to file
save_model: Save model file

# Gait Modes and Trial Types
gait_modes: Gait mode (i.e., subdirectories)
trials_use: Trials must include
trials_ignore: Trials can't include
label: Label name

# Subjects
subjects: List of subjects

# Input sensors
sensors: Input channels must include
sensors_ignore: Input channels can't include

# Network Training/Testing Parameters
num_epochs: Maximum number of epochs
min_epochs: Minimum number of epochs
steps_per_batch: Batch size
batch_size_per_step: Batch sequence length of label
early_stopping: Enable early stopping
patience: Early stopping patience
batch_pad_value: For padding batch if needed

# Network Hyperparameters
ksize: List of kernel sizes
hsize: List of channels sizes
levels: List of number of levels
loss: List of loss functions
opt: List of optimizers
dropout: List of dropout probabilities
lr: List of learning rates
pred: List of prediction times
eff_hist_limit: Maximum effecive history of model

# Run Details
enable_cuda: Enable the use of cuda
max_device_jobs: Maximum number of jobs to run on one device
'''

# Example configuration
config = config_util.Config()

config.input_dir = '../Data/*/STEP_CUSTOM_3'
config.output_dir = '../Results_test1/Default'

config.enable_cuda = False
config.max_device_jobs = 1

config.num_epochs = 500
config.gait_modes = ['LG', 'RA', 'RD', 'SA', 'SD']
config.subjects = ['AB03', 'AB04', 'AB05', 'AB06', 'AB07', 'AB09', 'AB11', 
				   'AB12', 'AB13', 'DS01', 'DS02', 'DS03', 'DS04', 'DS05']

# Ethan's edits
config.ksize_c = [4]
config.ksize_nc =[2]
config.levels_c = [5]
config.levels_nc = [3]
config.eff_hist_limit = 200
config.pred = [-14]