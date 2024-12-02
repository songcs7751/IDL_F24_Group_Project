import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import os
from biomechdata import TemporalConvNet
#from config_util import config
from config.config_util import config


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
        print(f"Input to forward pass: {x.shape}")

        # Check if the input length is sufficient for the effective history values
        if x.shape[2] <= max(self.eff_hist_c, self.eff_hist_nc):
            print(f"Input tensor length {x.shape[2]} is not sufficient for effective history values eff_hist_c: {self.eff_hist_c}, eff_hist_nc: {self.eff_hist_nc}")
            return torch.tensor([], dtype=torch.float32)

        # Slicing the tensor based on effective history values
        x_c = x[:, :, :self.eff_hist_nc + 1]  # Keeping the first `eff_hist_nc + 1` timesteps for causal TCN
        x_nc = x[:, :, -self.eff_hist_c - 1:]  # Keeping the last `eff_hist_c + 1` timesteps for non-causal TCN

        x_nc = torch.flip(x_nc, (2,))  # Flip for non-causal TCN

        print(f"x_c shape after slicing: {x_c.shape}")
        print(f"x_nc shape after slicing: {x_nc.shape}")

        x_c = self.tcn_causal(x_c)
        x_nc = self.tcn_noncausal(x_nc)

        if any(sequence_lens):
            y1_c = torch.cat([x_c[i, :, self.eff_hist_c:self.eff_hist_c + sequence_lens[i]].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
            y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:self.eff_hist_nc + sequence_lens[i]].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()
        else:
            y1_c = torch.cat([x_c[i, :, self.eff_hist_c:].contiguous() for i in range(x_c.shape[0])], dim=1).transpose(0, 1).contiguous()
            y1_nc = torch.cat([x_nc[i, :, self.eff_hist_nc:].contiguous() for i in range(x_nc.shape[0])], dim=1).transpose(0, 1).contiguous()

        print(f"y1_c shape before concatenation: {y1_c.shape}")
        print(f"y1_nc shape before concatenation: {y1_nc.shape}")

        y1 = torch.cat((y1_c, y1_nc), dim=1).contiguous()

        output = self.linear(y1)
        print(f"Output shape from model: {output.shape}")
        return output

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model_params = {
        'input_size_c': 20,
        'input_size_nc': 20,
        'output_size': 1,
        'num_channels_c': [50, 50, 50, 50, 50],
        'num_channels_nc': [50, 50, 50, 50, 50],
        'ksize_c': 4,
        'ksize_nc': 4,
        'dropout': 0.3,
        'eff_hist_c': 186, #자 이게 문제다. 왜 이 숫자로 지정되는지?
        'eff_hist_nc': 186
    }
    model = BilatTCN(**model_params)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    # sensors = [
    #     'hip_sagittal', 'd_hip_sagittal_lpf', 'thigh_accel_x', 'thigh_accel_y', 'thigh_accel_z',
    #     'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z',
    #     'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z'
    # ]

    sensors = ['Pelvis_V_ACCX', 'Pelvis_V_ACCY', 'Pelvis_V_ACCZ', 'Pelvis_V_GYROX', 'Pelvis_V_GYROY', 'Pelvis_V_GYROZ', 
			'LThigh_V_ACCX', 'LThigh_V_ACCY', 'LThigh_V_ACCZ', 'LThigh_V_GYROX', 'LThigh_V_GYROY', 'LThigh_V_GYROZ',
			'RThigh_V_ACCX', 'RThigh_V_ACCY', 'RThigh_V_ACCZ', 'RThigh_V_GYROX', 'RThigh_V_GYROY', 'RThigh_V_GYROZ',
			'hip_flexion_r', 'hip_flexion_l']

    # Check if required sensors are present in the DataFrame
    if not all(sensor in df.columns for sensor in sensors):
        print(f'Skipping {file_path} because required sensors are not in the columns.')
        return None, None

    # Drop rows with NaNs in the required columns
    df = df.dropna(subset=sensors + ['hip_flexion_l_moment'])

    X = df[sensors].values
    #'hip_flexion_l_moment'를 제외한 센서 데이터만 X에 할당
    # X = df[sensors].drop(columns=['hip_flexion_l_moment']).values

    y_true = df['hip_flexion_l_moment'].values
    return X, y_true

def predict(model, X):
    with torch.no_grad():
        y_pred = model(X).squeeze(0).numpy()
        print(f"Predicted output: {y_pred}")  # Debugging statement
    return y_pred

def calculate_metrics(y_true, y_pred):
    # Skip calculation if y_true or y_pred contains NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Skipping metric calculation due to NaN values in y_true or y_pred.")
        return float('nan'), float('nan')
    #rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse = np.sqrt(np.mean(y_true-y_pred)**2)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def sliding_window(X, window_size, step_size=1):
    num_sequences = (len(X) - window_size) // step_size + 1
    windows = np.array([X[i:i + window_size] for i in range(0, num_sequences, step_size)])
    return windows


def main():
    # model_path = '/Users/sunho/Desktop/TCN/Output/SavedModels/AB05_dropout0.3_hsize50_ksize_c4_ksize_nc4_levels_c5_levels_nc5_lossMSELoss_lr0.0005_optAdam_pred0.tar'
    #os.path.abspath(r"C:/Users/wlals/Downloads/ProcessedData")
    model_path = 'C:/Users/wlals/Downloads/ProcessedData/SavedModels/AB01_dropout0.3_hsize50_ksize_c4_ksize_nc4_levels_c5_levels_nc5_lossMSELoss_lr0.0005_optAdam_pred0.tar'
    model = load_model(model_path)

    data_dir = "C:/Users/wlals/Downloads/ProcessedData"
    results = []

    for subject in ['AB02']:
        for gait_mode in ['LG']:
            subject_dir = os.path.join(data_dir, subject, gait_mode)
            for trial in os.listdir(subject_dir):
                # Skip non-data files like .DS_Store
                if trial.startswith('.'):
                    continue

                if any(t in trial for t in config.trials_ignore):
                    continue
                
                trial_path = os.path.join(subject_dir, trial)
                print(f'Processing {trial_path}')
                
                X, y_true = load_data(trial_path)
                if X is None or y_true is None:
                    continue
                
                # print(f'Ground Truth Hip Moment is: ', y_true) # 이걸 통해서 y_true data를 뽑아오는건 정확한게 드러났음
                print(f'Input shape: {X.shape}, True labels shape: {y_true.shape}')
                # 여기서 Input shape: (4001, 21), True labels shape: (4001,) 이렇게 나옴

                # 94에서 model.eff_hist_c로 바꿨음
                if X.shape[0] < model.eff_hist_c:  # Minimum sequence length required 
                    print(f'Skipping {trial_path} due to insufficient input sequence length.')
                    continue

                # X_tensor = torch.tensor(X[:94].T, dtype=torch.float32).unsqueeze(0)  # Use only the first 94 timesteps
                # # 갑자기 왜 94? arbitrary인가 아니면 정해진것? 모델 트레이닝도 이렇게 진행된것?
                # print(f'X_tensor shape: {X_tensor.shape}')
                
                window_size = model.eff_hist_c+1
                step_size = 1                
                X_windows = sliding_window(X, window_size, step_size)
                print(f'Sliding window: {X_windows}')
                X_windows = X_windows.transpose(0, 2, 1)  # [num_sequences, 21, 94] 형태로 변환
                X_tensor = torch.tensor(X_windows, dtype=torch.float32)
                print(X_tensor.shape)

                y_pred = predict(model, X_tensor)
                print(f'Predicted labels shape: {y_pred.shape}')

                if y_pred.shape[0] == 0:
                    print('No predictions made. Skipping metric calculation.')
                    rmse, r2 = float('nan'), float('nan')
                else:
                    rmse, r2 = calculate_metrics(y_true[:y_pred.shape[0]], y_pred)

                results.append((trial_path, rmse, r2))
                print(f'Results for {trial_path}: RMSE: {rmse}, R²: {r2}')

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results, columns=['trial_path', 'rmse', 'r2'])
    # results_df.to_csv('/Users/sunho/Desktop/TCN/rmse_results.csv', index=False)
    #results_df.to_csv('C:/Users/wlals/Downloads/ProcessedData/TuningResults/rmse_results.csv', index=False)

    avg_rmse = results_df['rmse'].mean()
    avg_r2 = results_df['r2'].mean()

    # Create a DataFrame for the average row
    average_df = pd.DataFrame([['AVERAGE', avg_rmse, avg_r2]], columns=['trial_path', 'rmse', 'r2'])
    # Append the average row to the results DataFrame using pd.concat
    results_df = pd.concat([results_df, average_df], ignore_index=True)
    
    #data_dir = "C:/Users/wlals/Downloads/ProcessedData"
    result_dir = data_dir + '/TuningResults'
    result_file_name = subject + 'model_rmse_results.csv'
    # lc_df.to_csv(lc_file_name, index=False)
    results_df.to_csv(result_dir + '/' + result_file_name, index=False)


    for trial_path, rmse, r2 in results:
        print(f'Trial: {trial_path}, RMSE: {rmse}, R²: {r2}')
    
    print(f'AVERAGE RMSE: {avg_rmse}, R²: {avg_r2}')


if __name__ == '__main__':
    main()
