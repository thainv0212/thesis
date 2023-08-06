import sys
import time
sys.path.append('../')
from time import sleep
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field

import logging
from gmc_code.unsupervised.architectures.models.gmc import ICEGMC
from gmc_code.black_mamba import create_bm_name, BlackMambaSound

logger = logging.getLogger(__name__)
import torch
from blindai.trained_ai.fight_agent import SoundAgent as BlindAI


def check_args(args):
    for i in range(argc):
        if args[i] == "-n" or args[i] == "--n" or args[i] == "--number":
            global GAME_NUM
            GAME_NUM = int(args[i + 1])


def start_game(Character):
    model_paths = [
        # "D:\\final_with_delay\\real_new_delay_16000_epoch=33.pth", # the best one up to now 0.01 0.99

        # "final_with_delay\\real_new_delay_with_y_16000_epoch=24.pth",
        # "final_with_delay\\pretrain_sound_new_delay_16000last.pth"
        # 'final_with_delay\\real_new_delay_16000_0.03_0.97_normal_epoch=59.pth',
        # 'final_with_delay\\real_new_delay_16000_0.05_0.95_normal_epoch=22.pth',
        # 'final_with_delay\\real_new_delay_with_y_16000_0.03_0.97_normal_epoch=66.pth',
        # 'final_with_delay\\real_new_delay_with_y_16000_0.05_0.95_normal_epoch=50.pth',
        # 'final_with_delay\\real_new_delay_16000_with_y_pos_only_no_y_speed_0.01_0.99_no_speed_epoch=22.pth',
        # 'final_with_delay\\real_new_delay_16000_with_y_pos_only_no_y_speed_0.01_0.99_no_speed_epoch=50.pth',
        # 'final_with_delay\\pretrain_sound_winner_modified_no_y_16000_last.pth'
        # 'final_with_delay\\real_winner_modified_no_y_epoch=24.pth'
        # 'final_with_delay\\real_winner_modified_no_y_epoch=22.pth'
        # 'final_with_delay\\real_winner_modified_epoch=62.pth'
        # 'final_with_delay\\real_winner_modified_epoch=14.pth',
        # 'final_with_delay\\real_winner_original_epoch=21.pth'
        # 'final_with_delay\\pretrain_sound_winner_original_16000_last.pth'
        # 'final_with_delay\\real_new_delay_16000_normal_new_delay_30_30_epoch=27.pth'
        # 'D:\\final_with_delay\\real_new_delay_12000_mast_mast_normal_new_delay_20_15_12000_tiny_16000_32epoch=28.pth',
        # 'D:\\final_with_delay\\real_new_delay_16000_mast_mast_normal_new_delay_20_15_16000_tiny_16000_32epoch=28.pth',
        # 'D:\\final_with_delay\\real_new_delay_12000_passt_passt_normal_new_delay_20_15_12000_tiny224_16000_32epoch=11.pth',
        'D:\\final_with_delay\\real_new_delay_16000_passt_passt_normal_new_delay_20_15_16000_tiny224_16000_32epoch=63.pth',

    ]
    rates = [
        # '0.03_0.97',
        # '0.05_0.95',
        # '0.03_0.97',
        # '0.05_0.95',
        # '0_1'
        # '0.01_0.99',
        # '0.01_0.99',
        # '0.01_0.99',
        # '0.01_0.99',
        # '0.01_0.99',
        # '0.01_0.99',
        '0.01_0.99',
        # '0.01_0.99',
    ]

    designs = [
        # 'new',
        # 'new_delayed',
        # 'new_with_y',
        # 'new_with_y'
        # 'new_with_y_pos_only'
        # 'winner_modified_no_y',
        # 'winner_original'
        # 'winner_original_pretrain'
        # 'new_delayed',
        # 'new_delayed',
        'new_delayed',
        # 'new_delayed',
        # 'new_delayed',
    ]
    sound_lengths = [
        # 16000,
        # 16000,
        # 16000,
        # 16000,
        16000,
        # 12000,
        # 16000,
        # 12000,
        # 16000,
    ]
    active_mods = [
        # None,
        # None,
        # None,
        # None,
        # None
        # 'sound'
        # None
        # 'sound'
        # None,
        # None,
        None,
        # None,
        # None,
    ]
    frame_sizes = [
        # 30,
        # 30,
        # 30,
        # 30,
        # 26
        # 30,
        # 30,
        30,
        # 30,
        # 30,
    ]
    predict_types = [
        # 'normal',
        # 'normal',
        # 'normal',
        # 'normal',
        # 'no_y_speed',
        # 'no_y_speed',
        # 'normal',
        # 'normal',
        'normal',
        # 'normal',
        # 'normal',
    ]
    frame_sizes = {
        # 'normal': 30,
        # 'no_y_speed': 26
        # 'normal': 30,
        # 'normal': 30,
        'normal': 30,
        # 'normal': 30,
        # 'normal': 30,
    }
    transformer_types = [
        # 'ast',
        # 'mast',
        # 'mast',
        # 'passt',
        'passt'
    ]
    model_sizes = [
        # 'tiny224',
        # 'tiny',
        # 'tiny',
        # 'tiny224',
        'tiny224',
    ]
    # print(len(list(zip(
    #         model_paths, designs, sound_lengths, active_mods, rates, predict_types, transformer_types,
    #         model_sizes))))
    for model_ae_path, design, sound_length, active_mod, rate, predict_type, transformer_type, model_size in zip(
            model_paths, designs, sound_lengths, active_mods, rates, predict_types, transformer_types,
            model_sizes):

        for Chara in Character:
            # FFT GRU
            # load model
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device('cpu')
            frame_size = frame_sizes[predict_type]
            # device = torch.device('cuda')
            # sound_length = 24000
            common_dim = 20
            latent_dim = 15
            # frame_size = 30
            frame_size = frame_sizes[predict_type]
            num_mel_bins = 32
            sampling_rate = 16000
            n_frame = sound_length // 800
            # model_size = 'tiny224'
            # model_ae_path = 'final\\default_modalityNone_gmc_transformers_None_20_15_24000_tiny224_16000_32-FrameEncoder2-0.01-0.99-transition-1-0.01-0.99-5-0.01-[0, 1, 65, 66]_last.pth'
            model_bm_path = ''
            # design = 'default'
            model_params = {
                'name': 'model',
                'device': device,
                'common_dim': common_dim,
                'latent_dim': latent_dim,
                'frame_size': frame_size,
                'sound_length': sound_length,
                'num_mel_bins': num_mel_bins,
                'sampling_rate': sampling_rate,
                'device': device,
                'model_size': model_size,
                'model_ae_path': model_ae_path,
                'model_bm_path': model_bm_path,
                'n_frame': n_frame,
                'design': design,
                'active_mod': active_mod,
                'rate': rate,
                'predict_type': predict_type,
                'transformer_type': transformer_type,
            }
            # # mctsai65
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     bm_name = create_bm_name(**model_params)
            #     black_mamba = BlackMambaSound(gateway, ai_name=bm_name, **model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, bm_name, "MctsAi65", GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)

            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     black_mamba = BlackMambaSound(gateway, **model_params)
            #     bm_name = create_bm_name(**model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, "MctsAi65", bm_name, GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)

            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     black_mamba = BlackMambaSound(gateway, **model_params)
            #     bm_name = create_bm_name(**model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, "Sounder", bm_name, GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)

            # # BlindAI
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     black_mamba = BlackMambaSound(gateway, **model_params)
            #     bm_name = create_bm_name(**model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     rl_name = 'RLAI'
            #     manager.registerAI(rl_name,
            #                        BlindAI(gateway, logger=logger, encoder='mel', path='trained_model', rnn=True))
            #     game = manager.createGame(Chara, Chara, bm_name, rl_name, GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)

            # # mctsai65 MctsAi23i
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     bm_name = create_bm_name(**model_params)
            #     black_mamba = BlackMambaSound(gateway, ai_name=bm_name, **model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, bm_name, "MctsAi23i", GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)
            # Sounder
            black_mamba = BlackMambaSound(None, **model_params)

            start = time.time()
            for _ in range(100): black_mamba.warmup()
            end = time.time()
            print('warm up time', (end - start) / 100)
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     black_mamba = BlackMambaSound(gateway, **model_params)
            #     bm_name = create_bm_name(**model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, bm_name, "MctsAi23i", GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)
            #
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     black_mamba = BlackMambaSound(gateway, **model_params)
            #     bm_name = create_bm_name(**model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, bm_name, "Sounder", GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)

            # sounder
            # for _ in range(30):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     bm_name = create_bm_name(**model_params)
            #     black_mamba = BlackMambaSound(gateway, ai_name=bm_name, **model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, bm_name, "Sounder", GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)
            # for _ in range(5):
            #     gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
            #                           callback_server_parameters=CallbackServerParameters());
            #     manager = gateway.entry_point
            #     bm_name = create_bm_name(**model_params)
            #     black_mamba = BlackMambaSound(gateway, ai_name=bm_name, **model_params)
            #     manager.registerAI(bm_name, black_mamba)
            #     game = manager.createGame(Chara, Chara, "Sounder", bm_name, GAME_NUM)
            #     manager.runGame(game)
            #     # print(bm_name)
            #     close_gateway(gateway)


def close_gateway(g):
    g.close_callback_server()
    g.close()


def main_process(Chara):
    check_args(args)
    start_game(Chara)


path = ''
args = sys.argv
argc = len(args)
GAME_NUM = 1
Character = ["ZEN"]

main_process(Character)
