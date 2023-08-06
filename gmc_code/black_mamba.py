import sys

sys.path.append('./')
import numpy as np
import torch
# from avae3 import AVAE
# from vae_new import VAE
# from transforms import Transpose, Cutoff, Flatten, FFTTransform, MelTransform
import torchvision.transforms as T
import dill
from torch import nn
from gmc_code.unsupervised.architectures.models.gmc import ICEGMC
import torchaudio
import random
import os
import time

parent_dir = 'test_data'


def count_params(net):
    return sum(p.numel() for p in net.parameters())


if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)


def LeakyRelu(x):
    return (x >= 0) * x + 0.01 * (x < 0) * x


def Linear(x):
    return x


def Softmax(x):
    exp = np.exp(x)
    exp_sum = exp.sum()
    return exp / exp_sum


class Neural:
    weights = None
    biases = None
    activation = 'LeakyRelu'

    def __init__(self, weight_file, bias_file, activation):
        self.load_weights(weight_file)
        self.load_biases(bias_file)
        self.activation = activation

    def load_weights(self, file):
        weights = []
        with open(file, 'r') as f:
            data = f.readlines()
            for line in data:
                units = [float(f) for f in line.split(',')]
                weights.append(units)
        self.weights = np.array(weights)

    def load_biases(self, file):
        with open(file, 'r') as f:
            line = f.readline()
            units = [float(f) for f in line.split(',')]
            self.biases = np.array(units)

    def __call__(self, x):
        x_out = np.matmul(self.weights, x.T).reshape([self.weights.shape[0]]) + self.biases
        if self.activation == 'LeakyRelu':
            x_out = LeakyRelu(x_out)
        if self.activation == 'Softmax':
            x_out = Softmax(x_out)
        if self.activation == 'Linear':
            pass
        return x_out

    def generate_pytorch_layer(self):
        linear_layer = nn.Linear(*self.weights.shape)
        with torch.no_grad():
            linear_layer.weight.data = torch.tensor(self.weights)
            linear_layer.bias.data = torch.tensor(self.biases)
        if self.activation == 'LeakyRelu':
            activation = nn.LeakyReLU()
        elif self.activation == 'Softmax':
            activation = nn.Softmax()
        elif self.activation == 'Linear':
            activation = None
        if activation is None:
            return nn.Sequential(linear_layer)
        return nn.Sequential(linear_layer, activation)


class Network:
    neurals = []

    def __init__(self, neurals):
        self.neurals = neurals

    def __call__(self, x):
        for i in range(len(self.neurals)):
            x = self.neurals[i](x)
        return x


STATE_DIM = {
    1: {
        'raw': 160, 'fft': 512, 'mel': 2560
    },
    4: {
        'raw': 64, 'fft': 512, 'mel': 1280
    }
}


class BlackMambaSound:
    def __init__(self, gateway, **kwargs):
        # self.category_indice = {"project_tiles": [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
        #                         "positions": [2, 3, 67, 68], "speeds": [4, 5, 6, 7, 69, 70, 71, 72],
        #                         "actions": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        #                                     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        #                                     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 73,
        #                                     74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        #                                     93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        #                                     109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
        #                                     124, 125, 126, 127, 128], "energies": [1, 66], "hps": [0, 65],
        #                         "remaining_frames": [64, 129], "frame_nums": [130]}
        self.gateway = gateway
        # self.path = kwargs.get('model_path')
        # transform = kwargs.get('transform')
        # model = kwargs.get('model')
        # self.cats = kwargs.get('cats')
        # self.model = kwargs.get('model')
        # self.mask = kwargs.get('mask')
        self.masks = {
            'normal': [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 72, 129, 131, 132, 133, 134, 135, 136,
                       137,
                       138, 139, 140, 141, 142],
            'no_y_speed': [0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69, 70, 129, 131, 132, 133, 134, 135, 136, 137,
                           138, 139, 140, 141, 142]
        }
        self.predict_type = kwargs.get('predict_type')
        self.mask = self.masks[self.predict_type]

        # self.

        # self.actions = "AIR_A", "AIR_B", "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_DA", "AIR_DB", \
        #                "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "BACK_JUMP", "BACK_STEP", \
        #                "CROUCH_A", "CROUCH_B", "CROUCH_FA", "CROUCH_FB", "CROUCH_GUARD", "DASH", "FOR_JUMP", "FORWARD_WALK", \
        #                "JUMP", "NEUTRAL", "STAND_A", "STAND_B", "STAND_D_DB_BA", "STAND_D_DB_BB", "STAND_D_DF_FA", \
        #                "STAND_D_DF_FB", "STAND_D_DF_FC", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_FA", "STAND_FB", \
        #                "STAND_GUARD", "THROW_A", "THROW_B"

        self.actions = '''AIR,AIR_A,AIR_B,AIR_D_DB_BA,AIR_D_DB_BB,AIR_D_DF_FA,AIR_D_DF_FB,AIR_DA,AIR_DB,AIR_F_D_DFA,AIR_F_D_DFB,AIR_FA,AIR_FB,AIR_GUARD,AIR_GUARD_RECOV,AIR_RECOV,AIR_UA,AIR_UB,BACK_JUMP,BACK_STEP,CHANGE_DOWN,CROUCH,CROUCH_A,CROUCH_B,CROUCH_FA,CROUCH_FB,CROUCH_GUARD,CROUCH_GUARD_RECOV,CROUCH_RECOV,DASH,DOWN,FOR_JUMP,FORWARD_WALK,JUMP,LANDING,NEUTRAL,RISE,STAND,STAND_A,STAND_B,STAND_D_DB_BA,STAND_D_DB_BB,STAND_D_DF_FA,STAND_D_DF_FB,STAND_D_DF_FC,STAND_F_D_DFA,STAND_F_D_DFB,STAND_FA,STAND_FB,STAND_GUARD,STAND_GUARD_RECOV,STAND_RECOV,THROW_A,THROW_B,THROW_HIT,THROW_SUFFER'''.split(
            ',')
        # self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.device = kwargs.get('device')
        self.n_frame = kwargs.get('n_frame')
        self.num_mel_bins = kwargs.get('num_mel_bins')
        self.sampling_rate = kwargs.get('sampling_rate')
        self.down_sample = 48000 // self.sampling_rate
        # self.actor.get_init_state(self.device)
        self.round_count = 0
        self.frame_count = 0
        self.prev_action = None
        # vae
        # load black mamba
        # neurals = [
        #     Neural('BlackMamba/ZEN/normal/weight_l1.csv', 'BlackMamba/ZEN/normal/bias_l1.csv', activation='LeakyRelu'),
        #     Neural('BlackMamba/ZEN/normal/weight_l2.csv', 'BlackMamba/ZEN/normal/bias_l2.csv', activation='LeakyRelu'),
        #     Neural('BlackMamba/ZEN/normal/weight_l3.csv', 'BlackMamba/ZEN/normal/bias_l3.csv', activation='LeakyRelu'),
        #     Neural('BlackMamba/ZEN/normal/weight_l4.csv', 'BlackMamba/ZEN/normal/bias_l4.csv', activation='LeakyRelu'),
        #     Neural('BlackMamba/ZEN/normal/weight_l5.csv', 'BlackMamba/ZEN/normal/bias_l5.csv', activation='LeakyRelu'),
        #     Neural('BlackMamba/ZEN/normal/weight_l6.csv', 'BlackMamba/ZEN/normal/bias_l6.csv', activation='Softmax')]
        # blackmamba = Network(neurals)
        # network = Network(neurals)
        # pytorch_neurals = [n.generate_pytorch_layer() for n in neurals]
        # self.blackmamba_pytorch = nn.Sequential(*pytorch_neurals)
        # self.blackmamba_pytorch.eval()
        self.model, self.blackmamba_pytorch = create_blackmamba(**kwargs)
        print("param count: ",count_params(self.model), count_params(self.blackmamba_pytorch))
        self.model.eval()
        self.blackmamba_pytorch.eval()
        self.file_counter = 1
        self.game_data = []
        self.initial_values = {0: 1,
                               1: 0,
                               65: 1,
                               66: 0}
        self.prev_values = self.initial_values
        self.transition_index = [0, 1, 65, 66]
        self.frame_pairs = []
        self.frame_mask = [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 72, 129, 131, 132, 133, 134, 135,
                           136, 137,
                           138, 139, 140, 141, 142]
        # self.mask = [False] * 143
        # for cat in self.cats:
        #     for i in self.category_indice[cat]:
        #         self.mask[i] = True
        # self.mask = np.array(self.mask)
        # self.produced_indice = []

        # for cat in self.cats:
        #     self.produced_indice += self.category_indice[cat]
        self.tmp_state = torch.randn((2, 800 * self.n_frame), dtype=torch.float32)

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
        self.isGameJustStarted = True
        return 0

    def close(self):
        pass

    def getInformation(self, frameData, inControl):
        # Load the frame data every time getInformation gets called
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)
        # nonDelay = self.frameData
        # self.pre_framedata = self.nonDelay if self.nonDelay is not None else nonDelay
        # self.nonDelay = nonDelay
        self.isControl = inControl
        # self.currentFrameNum = nonDelay.getFramesNumber()  # first frame is 14

    def roundEnd(self, x, y, z):
        print(x)
        print(y)
        print(z)
        self.just_inited = True
        self.raw_audio_memory = None
        self.round_count += 1
        print('Finished {} round'.format(self.round_count))
        # dill.dump(self.game_data, open('game_data_{}.pkl'.format(self.file_counter), 'wb'))
        self.game_data = []
        self.file_counter += 1
        self.frame_count = 0
        # write frame data to file
        print('Writing frame data to file')
        now = int(time.time() * 1000)
        # dill.dump(self.frame_pairs, open(os.path.join(parent_dir, f'data_{now}.pkl'), 'wb'))
        self.frame_pairs = []

    def input(self):
        return self.inputKey

    @torch.no_grad()
    def processing(self):
        import time

        start = time.time()
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return
        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return
        self.inputKey.empty()
        self.cc.skillCancel()
        obs = self.raw_audio_memory
        if self.just_inited:
            self.just_inited = False
            # action_idx = np.random.choice(40, 1, replace=False)[0]
            action_idx = random.choice([32, 38])
        else:
            if obs is None:
                obs = np.zeros((2, 800 * self.n_frame))
            if not self.player:
                obs = np.array(obs[:, ::-1])
            state = torch.tensor(obs, dtype=torch.float32)
            fbank = torchaudio.compliance.kaldi.fbank(state[:, ::self.down_sample], htk_compat=True,
                                                      sample_frequency=self.sampling_rate,
                                                      use_energy=False,
                                                      window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0,
                                                      frame_shift=10)
            # produce frame data from sound
            fbank = fbank[None, :].to(self.device).type(torch.float32)
            # processed_frame = self.process_frame(self.frameData)['frame']
            # processed_frame = np.array(processed_frame)[self.frame_mask]
            # print(fbank.shape)
            # constructed_frame_elements = self.model.construct_frame(fbank, torch.tensor([processed_frame], dtype=torch.float32)).cpu().detach().numpy()
            constructed_frame_elements = self.model.construct_frame(fbank).cpu().detach().numpy()
            # reconstruct frame
            frame = np.zeros([1, 143])
            for i, v in zip(self.mask, constructed_frame_elements[0]):
                frame[0][i] = v
            # frame num
            frame[0][130] = self.frame_count / 3600
            # print(self.frame_count)

            # # add action info from the original frame
            # my = self.frameData.getCharacter(self.player)
            # opp = self.frameData.getCharacter(not self.player)
            # my_action = my.getAction().ordinal()
            # opp_action = opp.getAction().ordinal()
            # frame[0][8 + my_action] = 1
            # frame[1][73 + opp_action] = 1
            # # self previous action
            # if self.prev_action is not None:
            #     if self.player:
            #         start_idx = 8
            #     else:
            #         start_idx = 73
            #     frame[0][start_idx + self.prev_action] = 1

            if self.prev_action is not None:
                frame[0][8 + self.prev_action] = 1

            # # add Y position from frame
            # my = self.frameData.getCharacter(self.player)
            # opp = self.frameData.getCharacter(not self.player)
            # myY = (my.getBottom() + my.getTop()) / 2 / 640
            # oppY = (opp.getBottom() + opp.getTop()) / 2 / 640
            # frame[0][3] = myY
            # frame[0][68] = oppY
            # # # change X position
            # # myX = (my.getLeft() + my.getRight()) / 2 / 960
            # # oppX  = (opp.getLeft() + opp.getRight()) / 2 /960
            # # frame[0][2] = myX
            # # frame[0][67] = oppX
            # add speedY from frame
            # mySpeedY = (my.getSpeedY()) / 28
            # oppSpeedY = (opp.getSpeedY()) / 28
            # frame[0][6] = (mySpeedY >= 0)
            # frame[0][7] = np.abs(mySpeedY)
            # frame[0][71] = (oppSpeedY >= 0)
            # frame[0][72] = np.abs(oppSpeedY)
            # my_action = my.getAction().ordinal()
            # opp_action = opp.getAction().ordinal()
            # frame[0][8 + my_action] = 1
            # frame[0][73 + opp_action] = 1

            # frame[0][4] = (frame[0][4] >= 0.5)
            # frame[0][69] = (frame[0][69] >= 0.5)

            # add values based on transitions
            for (i, v) in self.prev_values.items():
                frame[0][i] = v + frame[0][i]
            # frame[0][6] = 0.5
            # frame[0][71] = 0.5
            # update previous frame values
            for i, v in self.prev_values.items():
                self.prev_values[i] = frame[0][i]
            frame = np.clip(frame, 0, 1)
            action_probs = self.blackmamba_pytorch(
                torch.from_numpy(frame).type(torch.float32).to(self.device)).cpu().detach().numpy()
            # real_frame = self.process_frame(self.frameData)['frame']
            # self.frame_pairs.append((frame, real_frame))
            # action_probs = self.blackmamba_pytorch(
            #     torch.from_numpy(np.array([self.process_frame(self.frameData)['frame']])).type(torch.float32).to(self.device)
            # ).cpu().detach().numpy()

            action_idx = np.argmax(action_probs)
            # print(action_probs)
            # print(action_idx)
            # print(action_idx)
        self.prev_action = action_idx
        self.cc.commandCall(self.actions[int(action_idx)])
        # print(self.actions[int(action_idx)], action_idx)
        self.inputKey = self.cc.getSkillKey()
        end = time.time()
        # print(end - start)

    def warmup(self):
        state = self.tmp_state
        for _ in range(1):
            fbank = torchaudio.compliance.kaldi.fbank(state[:, ::self.down_sample], htk_compat=True,
                                                      sample_frequency=self.sampling_rate,
                                                      use_energy=False,
                                                      window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0,
                                                      frame_shift=10)
            # produce frame data from sound
            fbank = fbank[None, :].to(self.device).type(torch.float32)
            # print(fbank.shape)
            constructed_frame_elements = self.model.construct_frame(fbank).cpu().detach().numpy()
            # reconstruct frame
            frame = np.zeros([1, 143])
            for i, v in zip(self.mask, constructed_frame_elements[0]):
                frame[0][i] = v
            # frame num
            frame[0][130] = self.frame_count / 3600

            if self.prev_action is not None:
                frame[0][8 + self.prev_action] = 1
            frame = np.clip(frame, 0, 1)

            action_probs = self.blackmamba_pytorch(
                torch.from_numpy(np.array([frame], np.float32)).to(self.device)
            ).cpu().detach().numpy()

            action_idx = np.argmax(action_probs)

    def getAudioData(self, audio_data):
        self.frame_count += 1
        self.audio_data = audio_data
        # process audio
        try:
            # start_time = time.time() * 1000
            byte_data = self.audio_data.getRawDataAsBytes()
            # end_time = time.time() * 1000
            np_array = np.frombuffer(byte_data, dtype=np.float32)
            np_array = np_array.reshape((2, 1024))
            # np_array = np_array.T
            # raw_audio = np_array[:800, :]
            raw_audio = np_array[:, :800]

            # self.logger.info('get data time {} {}'.format((end_time - start_time), raw_audio.sum()))
            # windows 1-2 ms
            # linux 40 ms
            # print('total time', end_time - start_time)
        except Exception as ex:
            # print('no audio')
            print(ex)
            raw_audio = np.zeros((2, 800))
            # raise ex # test
        if self.raw_audio_memory is None:
            print('raw_audio_memory none {}'.format(raw_audio.shape))
            # self.raw_audio_memory = np.expand_dims(raw_audio, axis=0)
            self.raw_audio_memory = raw_audio
        else:
            # self.raw_audio_memory = np.vstack((np.expand_dims(raw_audio, axis=0), self.raw_audio_memory))
            self.raw_audio_memory = np.hstack((self.raw_audio_memory, raw_audio))
            # self.raw_audio_memory = self.raw_audio_memory[:4, :, :]
            self.raw_audio_memory = self.raw_audio_memory[:, -800 * self.n_frame:]

        # append so that audio memory has the first shape of 4
        increase = (self.n_frame * 800 - self.raw_audio_memory.shape[1]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.hstack((np.zeros((2, 800)), self.raw_audio_memory))

    def getScreenData(self, sd):
        pass

    def process_frame(self, frame):
        input_list = []
        project_tiles = []
        positions = []
        speeds = []
        actions = []
        energies = []
        hps = []
        remaining_frames = []
        frame_nums = []

        project_tiles_idx = []
        positions_idx = []
        speeds_idx = []
        actions_idx = []
        energies_idx = []
        hps_idx = []
        remaining_frames_idx = []
        frame_nums_idx = []

        my = frame.getCharacter(self.player)
        opp = frame.getCharacter(not self.player)

        # my info
        myHp = np.abs(my.getHp() / 400)
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2) / 960
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 15
        mySpeedY = my.getSpeedY() / 28
        myState = my.getAction().ordinal()
        myRemainingFrame = my.getRemainingFrame() / 70

        # opp info
        oppHp = np.abs(opp.getHp() / 400)
        oppEnergy = opp.getEnergy() / 300
        oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = (opp.getSpeedX()) / 15
        oppSpeedY = (opp.getSpeedY()) / 28
        oppState = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        game_frame_num = frame.getFramesNumber() / 3600

        input_list.append(myHp)
        hps.append(myHp)
        hps_idx.append(len(input_list) - 1)
        input_list.append(myEnergy)
        energies.append(myEnergy)
        energies_idx.append(len(input_list) - 1)
        input_list.append(myX)
        positions_idx.append(len(input_list) - 1)
        input_list.append(myY)
        positions_idx.append(len(input_list) - 1)
        positions.append(myX)
        positions.append(myY)
        if mySpeedX < 0:
            input_list.append(0.0)
            speeds.append(0.0)
            speeds_idx.append(len(input_list) - 1)
        else:
            input_list.append(1.0)
            speeds.append(1.0)
            speeds_idx.append(len(input_list) - 1)
        input_list.append(np.abs(mySpeedX))
        speeds.append(np.abs(mySpeedX))
        speeds_idx.append(len(input_list) - 1)
        if mySpeedY < 0:
            input_list.append(0.0)
            speeds.append(0.0)
            speeds_idx.append(len(input_list) - 1)
        else:
            input_list.append(1.0)
            speeds.append(1.0)
            speeds_idx.append(len(input_list) - 1)
        input_list.append(np.abs(mySpeedY))
        speeds.append(np.abs(mySpeedY))
        speeds_idx.append(len(input_list) - 1)
        for i in range(0, 56):
            if (i == myState):
                input_list.append(1.0)
                actions.append(1.0)
                actions_idx.append(len(input_list) - 1)
            else:
                input_list.append(0.0)
                actions.append(0.0)
                actions_idx.append(len(input_list) - 1)
        input_list.append(myRemainingFrame)
        remaining_frames.append(myRemainingFrame)
        remaining_frames_idx.append(len(input_list) - 1)

        input_list.append(oppHp)
        hps.append(oppHp)
        hps_idx.append(len(input_list) - 1)
        input_list.append(oppEnergy)
        energies_idx.append(len(input_list) - 1)
        energies.append(oppEnergy)
        input_list.append(oppX)
        positions.append(oppX)
        positions_idx.append(len(input_list) - 1)
        input_list.append(oppY)
        positions.append(oppY)
        positions_idx.append(len(input_list) - 1)
        if oppSpeedX < 0:
            input_list.append(0.0)
            speeds.append(0.0)
            speeds_idx.append(len(input_list) - 1)
        else:
            input_list.append(1.0)
            speeds.append(1.0)
            speeds_idx.append(len(input_list) - 1)
        input_list.append(np.abs(oppSpeedX))
        speeds.append(np.abs(oppSpeedX))
        speeds_idx.append(len(input_list) - 1)
        if oppSpeedY < 0:
            input_list.append(0.0)
            speeds.append(0.0)
            speeds_idx.append(len(input_list) - 1)
        else:
            input_list.append(1.0)
            speeds.append(1.0)
            speeds_idx.append(len(input_list) - 1)
        input_list.append(np.abs(oppSpeedY))
        speeds.append(np.abs(oppSpeedY))
        speeds_idx.append(len(input_list) - 1)
        for i in range(0, 56):
            if i == oppState:
                input_list.append(1.0)
                actions.append(1.0)
                actions_idx.append(len(input_list) - 1)
            else:
                input_list.append(0.0)
                actions.append(0.0)
                actions_idx.append(len(input_list) - 1)

        input_list.append(oppRemainingFrame)
        remaining_frames.append(oppRemainingFrame)
        remaining_frames_idx.append(len(input_list) - 1)

        input_list.append(game_frame_num)
        frame_nums.append(game_frame_num)
        frame_nums_idx.append(len(input_list) - 1)

        myAttack = frame.getProjectilesByP1() if self.player else frame.getProjectilesByP2()
        oppAttack = frame.getProjectilesByP2() if self.player else frame.getProjectilesByP1()

        for i in range(0, 2):
            if myAttack.size() > i:
                tmp = myAttack.get(i)
                input_list.append(tmp.getHitDamage() / 200.0)
                project_tiles.append(tmp.getHitDamage() / 200.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(
                    ((tmp.getCurrentHitArea().getLeft() + tmp.getCurrentHitArea().getRight()) / 2) / 960.0
                )
                project_tiles.append(
                    ((tmp.getCurrentHitArea().getLeft() + tmp.getCurrentHitArea().getRight()) / 2) / 960.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(
                    ((tmp.getCurrentHitArea().getTop() + tmp.getCurrentHitArea().getBottom()) / 2) / 640.0
                )
                project_tiles_idx.append(len(input_list) - 1)
                project_tiles.append(
                    ((tmp.getCurrentHitArea().getTop() + tmp.getCurrentHitArea().getBottom()) / 2) / 640.0)

            else:
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                project_tiles.append(0.0)
                project_tiles.append(0.0)
                project_tiles.append(0.0)

        for i in range(0, 2):
            if oppAttack.size() > i:
                tmp = oppAttack.get(i)
                input_list.append(tmp.getHitDamage() / 200.0)
                project_tiles.append(tmp.getHitDamage() / 200.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(
                    ((tmp.getCurrentHitArea().getLeft() + tmp.getCurrentHitArea().getRight()) / 2) / 960.0
                )
                project_tiles.append(
                    ((tmp.getCurrentHitArea().getLeft() + tmp.getCurrentHitArea().getRight()) / 2) / 960.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(
                    ((tmp.getCurrentHitArea().getTop() + tmp.getCurrentHitArea().getBottom()) / 2) / 640.0
                )
                project_tiles.append(
                    ((tmp.getCurrentHitArea().getTop() + tmp.getCurrentHitArea().getBottom()) / 2) / 640.0)
                project_tiles_idx.append(len(input_list) - 1)

            else:
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                input_list.append(0.0)
                project_tiles_idx.append(len(input_list) - 1)
                project_tiles.append(0.0)
                project_tiles.append(0.0)
                project_tiles.append(0.0)

        for t in input_list, project_tiles, positions, speeds, actions, energies, hps, remaining_frames, frame_nums:
            for i in range(0, len(t)):
                if t[i] > 1.0:
                    t[i] = 1
                if t[i] < 0.0:
                    t[i] = 0
        return {'frame': input_list,
                'project_tiles': (project_tiles, project_tiles_idx),
                'positions': (positions, positions_idx),
                'speeds': (speeds, speeds_idx), \
                'actions': (actions, actions_idx),
                'energies': (energies, energies_idx),
                'hps': (hps, hps_idx),
                'remaining_frames': (remaining_frames, remaining_frames_idx), \
                'frame_nums': (frame_nums, frame_nums_idx)}

    class Java:
        implements = ["aiinterface.AIInterface"]


def create_bm_name(**kwargs):
    model_name = kwargs.get('name')
    common_dim = kwargs.get('common_dim')
    latent_dim = kwargs.get('latent_dim')
    frame_size = kwargs.get('frame_size')
    sound_length = kwargs.get('sound_length')
    num_mel_bins = kwargs.get('num_mel_bins')
    sampling_rate = kwargs.get('sampling_rate')
    model_size = kwargs.get('model_size')
    design = kwargs.get('design')
    active_mod = kwargs.get('active_mod')
    rate = kwargs.get('rate')
    predict_type = kwargs.get('predict_type')
    transformer_type = kwargs.get('transformer_type')
    name = f'BlindBlackMamba_{transformer_type}_{predict_type}_{active_mod}_{design}_{model_name}_{common_dim}_{latent_dim}_{frame_size}_{sound_length}_{num_mel_bins}_{sampling_rate}_{model_size}_{rate}_'
    return name


def create_blackmamba(**kwargs):
    model_name = kwargs.get('name')
    common_dim = kwargs.get('common_dim')
    latent_dim = kwargs.get('latent_dim')
    frame_size = kwargs.get('frame_size')
    sound_length = kwargs.get('sound_length')
    num_mel_bins = kwargs.get('num_mel_bins')
    sampling_rate = kwargs.get('sampling_rate')
    model_size = kwargs.get('model_size')
    device = kwargs.get('device')
    autoencoder_path = kwargs.get('model_ae_path')
    head_path = kwargs.get('model_bm_path')
    active_mod = kwargs.get('active_mod')
    predict_type = kwargs.get('predict_type')
    transformer_type = kwargs.get('transformer_type')
    # load autoencoder
    autoencoder = ICEGMC(name=model_name, common_dim=common_dim, latent_dim=latent_dim, frame_size=frame_size,
                         sound_length=sound_length, num_mel_bins=num_mel_bins, sampling_rate=sampling_rate,
                         model_size=model_size, frame_encoder=2, active_mod=active_mod,
                         transformer_type=transformer_type).to(device)
    autoencoder.load_checkpoint(autoencoder_path)
    autoencoder.eval()
    # blackmamba
    neurals = [
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l1.csv', head_path + 'BlackMamba/ZEN/normal/bias_l1.csv',
               activation='LeakyRelu'),
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l2.csv', head_path + 'BlackMamba/ZEN/normal/bias_l2.csv',
               activation='LeakyRelu'),
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l3.csv', head_path + 'BlackMamba/ZEN/normal/bias_l3.csv',
               activation='LeakyRelu'),
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l4.csv', head_path + 'BlackMamba/ZEN/normal/bias_l4.csv',
               activation='LeakyRelu'),
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l5.csv', head_path + 'BlackMamba/ZEN/normal/bias_l5.csv',
               activation='LeakyRelu'),
        Neural(head_path + 'BlackMamba/ZEN/normal/weight_l6.csv', head_path + 'BlackMamba/ZEN/normal/bias_l6.csv',
               activation='Softmax')
    ]
    # network = Network(neurals)
    pytorch_neurals = [n.generate_pytorch_layer() for n in neurals]
    blackmamba_pytorch = nn.Sequential(*pytorch_neurals).to(device).float()
    blackmamba_pytorch.eval().float()
    return autoencoder, blackmamba_pytorch


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # mask = [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 72, 129, 131, 132, 133, 134, 135, 136, 137,
    #         138, 139, 140, 141, 142]
    # audio = torch.randn((1, 48, 128)).to(device)
    # autoencoder = ICEGMC(name='model', common_dim=20, latent_dim=15, frame_size=30, sound_length=24000,
    #                      num_mel_bins=128,
    #                      sampling_rate=48000,
    #                      model_size='tiny224').to(device)
    # autoencoder.load_checkpoint(
    #     '/home/thai/gmc_transformers_None_20_15_24000_tiny224-epoch=00.pth')
    # autoencoder.eval()
    # parent_dir = '/mnt/DATA/OneDrive/Ritsumei/Laboratory/Competitions/updated_2021Competition/AIs/BlackMamba/'
    # neurals = [
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l1.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l1.csv',
    #            activation='LeakyRelu'),
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l2.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l2.csv',
    #            activation='LeakyRelu'),
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l3.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l3.csv',
    #            activation='LeakyRelu'),
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l4.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l4.csv',
    #            activation='LeakyRelu'),
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l5.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l5.csv',
    #            activation='LeakyRelu'),
    #     Neural(parent_dir + 'BlackMamba/ZEN/normal/weight_l6.csv', parent_dir + 'BlackMamba/ZEN/normal/bias_l6.csv',
    #            activation='Softmax')
    # ]
    # network = Network(neurals)
    # pytorch_neurals = [n.generate_pytorch_layer() for n in neurals]
    # blackmamba_pytorch = nn.Sequential(*pytorch_neurals).to(device)
    # blackmamba_pytorch.eval()
    # data = np.random.rand(1, 143) - 0.5
    # import time
    # import numpy as np
    #
    # start = time.time()
    # for _ in range(1000):
    #     constructed_frame_elements = autoencoder.construct_frame(audio).detach().numpy()
    #     frame = np.zeros([1, 143])
    #     for i, v in zip(mask, constructed_frame_elements[0]):
    #         frame[0][i] = v
    #     probs = blackmamba_pytorch(torch.Tensor(frame).type(torch.float64).to(device)).detach().numpy()
    #     # print(probs)
    #     end = time.time()
    # print((end - start) / 1000)

    # print(pytorch_neurals(torch.Tensor(data).type(torch.float64)).shape)
    # import time
    # start = time.time_ns()
    # for _ in range(10000):
    #     network(data).shape
    # end = time.time_ns()
    # print((end-start)/10000/1e9)
    # start = time.time_ns()
    # for _ in range(10000):
    #     with torch.no_grad():
    #         pytorch_neurals(torch.Tensor(data).type(torch.float64)).shape
    # end = time.time_ns()
    # print((end-start)/10000/1e9)
    raw_audio = np.ones((2, 800))
    raw_audio_memory = None
    raw_audio_memory = np.ones((2, 3200))
    n_frame = 4
    if raw_audio_memory is None:
        print('raw_audio_memory none {}'.format(raw_audio.shape))
        # self.raw_audio_memory = np.expand_dims(raw_audio, axis=0)
        raw_audio_memory = raw_audio
    else:
        # self.raw_audio_memory = np.vstack((np.expand_dims(raw_audio, axis=0), self.raw_audio_memory))
        raw_audio_memory = np.hstack((raw_audio_memory, raw_audio))
        # self.raw_audio_memory = self.raw_audio_memory[:4, :, :]
        raw_audio_memory = raw_audio_memory[:, -800 * n_frame:]

    # append so that audio memory has the first shape of 4
    increase = (n_frame * 800 - raw_audio_memory.shape[1]) // 800
    for _ in range(increase):
        raw_audio_memory = np.hstack((np.zeros((2, 800), raw_audio_memory)))
    print(raw_audio_memory.shape)
