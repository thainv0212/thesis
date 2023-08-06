import os
import json
import pickle
import numpy as np
import lzma
from tqdm import tqdm


def create_empty_data(data):
    new_data = {}
    for k, v in data.items():
        new_data[k] = np.zeros_like(v)
    return new_data



path = '/mnt/DATA/OneDrive/Ritsumei/Laboratory/Research/gmc_transformer/sample'
dst_path = '/home/thai/result_test'


def extract_data(path, dst_path):
    data_files = os.listdir(path)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    file_idx = {}
    count = 0
    sequence_len = 10
    for file in tqdm(data_files):
        data = json.load(open(os.path.join(path, file), 'r'))
        for r in data:
            # cutoff sound data
            r['Audio'] = np.array(r['rawAudio'])[:, :800]
        for i in range(len(data)):
            # audio_data = [u['Audio'] for u in data[:i]]
            # audio_data = [np.zeros((2, 800))] * (sequence_len - len(audio_data)) + audio_data
            # audio_data = np.hstack(audio_data)
            audio_data = data[i]['Audio']
            frame = data[i]['frameDataDelay']
            s = {'Audio': audio_data, 'Frame': frame, 'File': file, 'Idx': i}
            # save data
            pickle.dump(s, open(os.path.join(dst_path, f'{count}.pkl'), 'wb'))
            count += 1
# # better design
# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design/test',
#              '/mnt/SAMSUNG870/Thai/Data/better_design/test_extract_delay')
# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design/train',
#              '/mnt/SAMSUNG870/Thai/Data/better_design/train_extract_delay')
#
# # default
# extract_data('/home/dl-station/Thai/VAE/sound_data_recollect/repaired/train',
#              '/home/dl-station/Thai/VAE/sound_data_recollect/repaired/train_extract_delay')
# extract_data('/home/dl-station/Thai/VAE/sound_data_recollect/repaired/test',
#              '/home/dl-station/Thai/VAE/sound_data_recollect/repaired/test_extract_delay')
#
# # default
# extract_data('/mnt/SAMSUNG870/Thai/Data/winner_design/test',
#              '/mnt/SAMSUNG870/Thai/Data/winner_design/test_extract_delay')
# extract_data('/mnt/SAMSUNG870/Thai/Data/winner_design/train',
#              '/mnt/SAMSUNG870/Thai/Data/winner_design/train_extract_delay')

# better design with Y
# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/test',
#              '/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/test_extract_delay')
# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/train',
#              '/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/train_extract_delay')

# better design with Y position only

# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/train',
#              '/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/train_extract_delay')
# extract_data('/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/test',
#              '/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/test_extract_delay')

# better design with Y position only (no bgm)
# extract_data('/home/thai/Downloads/PC10_Y_position_only_nobgm_train',
#              '/home/thai/Downloads/PC10_Y_position_only_nobgm_train_extract_delay')
# extract_data('/home/thai/Downloads/PC10_Y_position_only_nobgm_test',
#              '/home/thai/Downloads/PC10_Y_position_only_nobgm_test_extract_delay')