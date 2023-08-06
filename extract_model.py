from gmc_code.unsupervised.architectures.models.gmc import ICEGMC
import torch
import os

parent_dir = 'pretrained/20230502'
model_ae_paths = [
    # 'pretrain_frame_new_delay_with_y_new_delay_with_y_modalityframe_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth',
    # 'pretrain_sound_new_delay_with_y_new_delay_with_y_modalitysound_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth'
    # "pretrain_frame_new_delay_with_y_pos_only_no_y_speed_new_delay_with_y_pos_only_modalityframe_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth",
    # "pretrain_sound_new_delay_with_y_pos_only_no_y_speed_new_delay_with_y_pos_only_modalitysound_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth",
    # "pretrain_frame_new_delay_with_y_pos_only_nobgm_no_y_speed_new_delay_with_y_pos_only_nobgm_modalityframe_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth",
    "pretrain_sound_new_delay_with_y_pos_only_nobgm_no_y_speed_new_delay_with_y_pos_only_nobgm_modalitysound_gmc_transformers_None_20_15_16000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]epoch=50.pth",

]
designs = [
    # 'with_y_position_no_y_speed',
    # 'with_y_position_no_y_speed',
    # 'with_y_position_no_y_speed_nobgm',
    'with_y_position_no_y_speed_nobgm',
]
modalities = [
    # 'frame',
    # 'sound',
    # 'frame',
    'sound'
]
lengths = [
    # 16000,
    # 16000,
    # 16000,
    16000,
]


def extract_model(path, modality, design, sound_length):
    # sound_length = 24000
    common_dim = 20
    latent_dim = 15
    frame_size = 26
    num_mel_bins = 32
    sampling_rate = 16000
    n_frame = sound_length // 800
    model_size = 'tiny224'
    # model_ae_path = 'gmc_transformers_None_20_15_24000_tiny224_16000_16-epoch=05.pth'
    # model_ae_path = '/home/thai/Desktop/pretrained/winner_modalitysound_gmc_transformers_None_20_15_24000_tiny224_16000_32-FrameEncoder2-0-1-transition-1-0-1-5-0.01-[0, 1, 65, 66]_last.pth'
    autoencoder = ICEGMC(name='model', common_dim=common_dim, latent_dim=latent_dim, frame_size=frame_size,
                         sound_length=sound_length, num_mel_bins=num_mel_bins, sampling_rate=sampling_rate,
                         model_size=model_size, frame_encoder=2, active_mod=modality, transformer_type='ast')
    autoencoder.load_checkpoint(os.path.join(parent_dir, path))
    autoencoder.eval()
    model = None
    if modality == 'sound':
        model = autoencoder.sound_processor.sound_features
    elif modality == 'frame':
        model = autoencoder.frame_processor.frame_features
    else:
        raise Exception('No valid exception')

    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(parent_dir, f"{design}_{modality}_{sound_length}_model.pth.tar")
    )


for (path, modality, design, sound_length) in zip(model_ae_paths, modalities, designs, lengths):
    extract_model(path, modality, design, sound_length)
