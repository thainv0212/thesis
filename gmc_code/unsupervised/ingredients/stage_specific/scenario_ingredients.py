import sacred

########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")


@scenario_ingredient.named_config
def mhd():
    scenario = "mhd"
    data_dir = "./dataset/"


@scenario_ingredient.named_config
def ice():
    name = 'autoencoder'
    scenario = 'ice'
    dataset = 'new'
    data_dir = {
        'new': ['/mnt/SAMSUNG870/Thai/Data/better_design/train_extract',
                '/mnt/SAMSUNG870/Thai/Data/better_design/test_extract'],
        'default': ['/home/dl-station/Thai/VAE/sound_data_recollect/extracted/train',
                    '/home/dl-station/Thai/VAE/sound_data_recollect/extracted/test'],
        'winner': ['/mnt/SAMSUNG870/Thai/Data/winner_design/train_extract',
                   '/mnt/SAMSUNG870/Thai/Data/winner_design/test_extract'],

        'new_delay': ['/mnt/SAMSUNG870/Thai/Data/better_design/train_extract_delay',
                      '/mnt/SAMSUNG870/Thai/Data/better_design/test_extract_delay'],
        'default_delay': ['/home/dl-station/Thai/VAE/sound_data_recollect/repaired/train_extract_delay',
                          '/home/dl-station/Thai/VAE/sound_data_recollect/repaired/test_extract_delay'],
        'winner_delay': ['/mnt/SAMSUNG870/Thai/Data/winner_design/train_extract_delay',
                         '/mnt/SAMSUNG870/Thai/Data/winner_design/test_extract_delay'],
        'new_delay_with_y': ['/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/train_extract_delay',
                             '/mnt/SAMSUNG870/Thai/Data/better_design_with_Y/test_extract_delay'],
        'new_delay_with_y_pos_only_nobgm': [
            '/home/thai/Downloads/PC10_Y_position_only_nobgm_train_extract_delay',
            '/home/thai/Downloads/PC10_Y_position_only_nobgm_test_extract_delay'],
        'new_delay_with_y_pos_only': ['/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/train_extract_delay',
                                      '/mnt/SAMSUNG870/Thai/Data/better_design_Y_position_only/test_extract_delay']
    }
    frame_mask = [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 72, 129, 131, 132, 133, 134, 135, 136, 137,
                  138, 139, 140, 141, 142]
    frame_mask = {
        'normal': [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 72, 129, 131, 132, 133, 134, 135, 136, 137,
                   138, 139, 140, 141, 142],
        'no_y_speed': [0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69, 70, 129, 131, 132, 133, 134, 135, 136, 137,
                       138, 139, 140, 141, 142]
    }
    predict_type = 'normal'
    sampling_rate = 16000
    transition_mask = [0, 1, 65, 66]
