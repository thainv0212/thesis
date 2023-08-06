import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from gmc_code.unsupervised.architectures.models.model import TransformerEncoder, FrameEncoder, FrameDecoder, \
    FrameEncoder1, FrameEncoder2, FrameEncoder3, FrameDecoder1, FrameDecoder2, FrameDecoder3


def load_checkpoint(model, model_file, freeze=True):
    checkpoint = torch.load(model_file)['state_dict']
    for key in list(checkpoint.keys()):
        if 'model.' in key:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
        model.load_state_dict(checkpoint)
    if freeze:
        for (_, para) in model.named_parameters():
            para.requires_grad = False
    return model


def create_frame_encoder(encoder_type, frame_size, common_dim):
    if encoder_type == 0:
        print('use FrameEncoder0')
        return FrameEncoder(frame_size, common_dim)
    if encoder_type == 1:
        print('use FrameEncoder1')
        return FrameEncoder1(frame_size, common_dim)
    if encoder_type == 2:
        print('use FrameEncoder2')
        return FrameEncoder2(frame_size, common_dim)
    if encoder_type == 3:
        print('use FrameEncoder3')
        return FrameEncoder3(frame_size, common_dim)
    raise Exception("No valid encoder")


def create_frame_decoder(decoder_type, latent_dim, output_dim):
    # self.frame_decoder = FrameDecoder(latent_dim, output_dim)
    if decoder_type == 0:
        print('use FrameDecoder0')
        return FrameDecoder(latent_dim, output_dim)
    elif decoder_type == 1:
        print('use FrameDecoder1')
        return FrameDecoder1(latent_dim, output_dim)
    elif decoder_type == 2:
        print('use FrameDecoder2')
        return FrameDecoder2(latent_dim, output_dim)
    elif decoder_type == 3:
        print('use FrameDecoder3')
        return FrameDecoder3(latent_dim, output_dim)
    else:
        raise Exception("No frame decoder")


class MHDCommonEncoder(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class MHDImageProcessor(nn.Module):

    def __init__(self, common_dim):
        super(MHDImageProcessor, self).__init__()
        self.common_dim = common_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDSoundProcessor, self).__init__()

        # Properties
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.projector = nn.Linear(2048, common_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDTrajectoryProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDTrajectoryProcessor, self).__init__()

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        # Output layer of the network
        self.projector = nn.Linear(512, common_dim)

    def forward(self, x):
        h = self.trajectory_features(x)
        return self.projector(h)


class MHDLabelProcessor(nn.Module):

    def __init__(self, common_dim):
        super(MHDLabelProcessor, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(10, common_dim)

    def forward(self, x):
        return self.projector(x)


class MHDJointProcessor(nn.Module):
    """
    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, common_dim):
        super(MHDJointProcessor, self).__init__()
        self.common_dim = common_dim

        # Modality-specific features
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )

        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.projector = nn.Linear(128 * 7 * 7 + 2048 + 512 + 10, common_dim)

    def forward(self, x):
        x_img, x_sound, x_trajectory, x_label = x[0], x[1], x[2], x[3]

        # Image
        h_img = self.img_features(x_img)
        h_img = h_img.view(h_img.size(0), -1)

        # Sound
        h_sound = self.sound_features(x_sound)
        h_sound = h_sound.view(h_sound.size(0), -1)

        # Trajectory
        h_trajectory = self.trajectory_features(x_trajectory)

        return self.projector(torch.cat((h_img, h_sound, h_trajectory, x_label), dim=-1))


"""


Extra components


"""


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# FightingICE

# class ICECommonEncoder(nn.Module):
#
#     def __init__(self, common_dim, latent_dim):
#         super(MHDCommonEncoder, self).__init__()
#         self.common_dim = common_dim
#         self.latent_dim = latent_dim
#
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(common_dim, 512),
#             Swish(),
#             nn.Linear(512, 512),
#             Swish(),
#             nn.Linear(512, latent_dim),
#         )
#
#     def forward(self, x):
#         return F.normalize(self.feature_extractor(x), dim=-1)


class ICEFrameProcessor(nn.Module):
    def __init__(self, frame_size, common_dim, encoder_type, frame_model_pretrain):
        super(ICEFrameProcessor, self).__init__()

        # self.frame_features = FrameEncoder(frame_size, common_dim)
        # if encoder_type == 0:
        #     self.frame_features = FrameEncoder(frame_size, common_dim)
        # elif encoder_type == 1:
        #     self.frame_features = FrameEncoder1(frame_size, common_dim)
        # elif encoder_type == 2:
        #     self.frame_features = FrameEncoder2(frame_size, common_dim)
        self.frame_features = create_frame_encoder(encoder_type, frame_size, common_dim)
        if frame_model_pretrain is not None:
            print('Load frame model from pretrained model')
            self.frame_features = load_checkpoint(self.frame_features, frame_model_pretrain)

        # Output layer of the network
        self.projector = nn.Linear(common_dim, common_dim)

    def forward(self, x):
        h = self.frame_features(x)
        return self.projector(h)


class ICESoundProcessor(nn.Module):
    def __init__(self, sound_data_length, common_dim, num_mel_bins=128, sampling_rate=48000, model_size=None,
                 sound_model_pretrain=None, transformer_type=None):
        super(ICESoundProcessor, self).__init__()
        self.sound_data_length = sound_data_length
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        # Properties
        self.sound_features = TransformerEncoder(data_size=sound_data_length, num_mel_bins=num_mel_bins,
                                                 sampling_rate=sampling_rate, model_size=model_size,
                                                 transformer=transformer_type)
        if sound_model_pretrain is not None:
            print('Load sound model from pretrained model', sound_model_pretrain)
            self.sound_features = load_checkpoint(self.sound_features, sound_model_pretrain)

        # Output layer of the network
        self.projector = nn.Linear(self.calculate_size(), common_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h.type(torch.float32))

    def calculate_size(self):
        sound_input = torch.randn((2, self.sound_data_length))
        down_sample = 48000 // self.sampling_rate
        fbank = torchaudio.compliance.kaldi.fbank(sound_input[:, ::down_sample], htk_compat=True,
                                                  sample_frequency=self.sampling_rate,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0,
                                                  frame_shift=10)
        sound_output = self.sound_features(fbank[None, :, :])
        if isinstance(sound_output, tuple):
            sound_output = sound_output[1]
        return sound_output.size().numel()


class ICEFrameReconstructor(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_type):
        super(ICEFrameReconstructor, self).__init__()
        # self.frame_decoder = FrameDecoder(latent_dim, output_dim)
        # if decoder_type == 0:
        #     self.frame_decoder = FrameDecoder(latent_dim, output_dim)
        # elif decoder_type == 1:
        #     self.frame_decoder = FrameDecoder1(latent_dim, output_dim)
        # elif decoder_type == 2:
        #     self.frame_decoder = FrameDecoder2(latent_dim, output_dim)
        # else:
        #     raise Exception("No frame decoder")
        self.frame_decoder = create_frame_decoder(decoder_type, latent_dim, output_dim)

    def forward(self, x):
        h = self.frame_decoder(x)
        h = h.view(h.size(0), -1)
        return h


class ICEJointProcessor(nn.Module):
    """
    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, common_dim, sound_data_length, frame_size, num_mel_bins, sampling_rate=48000, model_size=None,
                 frame_encoder_type=None, frame_model_pretrain=None, sound_model_pretrain=None, transformer_type=None):
        super(ICEJointProcessor, self).__init__()
        print('Initialize Joint processor:')
        self.common_dim = common_dim
        self.sound_data_length = sound_data_length
        self.frame_size = frame_size
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.frame_encoder_type = frame_encoder_type

        self.sound_features = TransformerEncoder(data_size=sound_data_length, num_mel_bins=num_mel_bins,
                                                 sampling_rate=sampling_rate, model_size=model_size,
                                                 transformer=transformer_type)
        if sound_model_pretrain is not None:
            print('Load pretrained sound model')
            self.sound_features = load_checkpoint(self.sound_features, sound_model_pretrain)
        # self.frame_features = FrameEncoder(frame_size, common_dim)
        # if frame_encoder_type == 0:
        #     self.frame_features = FrameEncoder(frame_size, common_dim)
        # elif frame_encoder_type == 1:
        #     self.frame_features = FrameEncoder1(frame_size, common_dim)
        # elif frame_encoder_type == 2:
        #     self.frame_features = FrameEncoder2(frame_size, common_dim)
        # else:
        #     raise Exception("There is no frame encoder")
        self.frame_features = create_frame_encoder(frame_encoder_type, frame_size, common_dim)
        if frame_model_pretrain is not None:
            print('Load pretrained frame model')
            self.frame_features = load_checkpoint(self.frame_features, frame_model_pretrain)

        self.projector = nn.Linear(self.calculate_size(), common_dim)

    def forward(self, x):
        x_sound, x_frame = x[0], x[1]

        # Frame
        h_frame = self.frame_features(x_frame)
        h_frame = h_frame.view(h_frame.size(0), -1)

        # Sound
        h_sound = self.sound_features(x_sound)
        h_sound = h_sound.view(h_sound.size(0), -1)

        return self.projector(torch.cat((h_sound, h_frame), dim=-1))

    def calculate_size(self):
        # sound
        # sound_input = torch.randn((1, self.sound_data_length, 2))
        # sound_output = self.sound_features(sound_input)

        sound_input = torch.randn((2, self.sound_data_length))
        down_sample = 48000 // self.sampling_rate
        fbank = torchaudio.compliance.kaldi.fbank(sound_input[:, ::down_sample], htk_compat=True,
                                                  sample_frequency=self.sampling_rate,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0,
                                                  frame_shift=10)
        sound_output = self.sound_features(fbank[None, :, :])

        # frame
        frame_input = torch.randn((1, self.frame_size))
        # frame_output = self.frame_features(frame_input)
        temp = create_frame_encoder(self.frame_encoder_type, self.frame_size, self.common_dim)
        temp.eval()
        frame_output = temp(frame_input)
        # print(sound_output.shape, frame_output.shape)
        return sound_output.size().numel() + frame_output.size().numel()


class ICECommonEncoder(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(ICECommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Linear(64, latent_dim),
            Swish()
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


if __name__ == '__main__':
    ICEJointProcessor(15, 8000, 29, 32, model_size='tiny224')
