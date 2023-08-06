from abc import ABC, abstractmethod
import torch
import torchaudio.transforms
import torchaudio
from torch import nn
from torch.cuda.amp import autocast
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import os
import wget
from gmc_code.unsupervised.architectures.models.passt import get_model


class BaseSoundEncoder(nn.Module, ABC):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(BaseSoundEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.frame_skip = frame_skip

    def forward(self, x):
        # left side
        left = x[:, :, 0]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, :, 1]
        right = self.encode_single_channel(right)
        return torch.cat((left, right), dim=1)

    @abstractmethod
    def encode_single_channel(self, data):
        pass


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False,
                 audioset_pretrain=False, model_size='tiny224', verbose=True, patch_size=8):

        super(ASTModel, self).__init__()
        # assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),
                                                                                  str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                # self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain) # 0.4.5
                self.v = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                # self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain) # 0.4.5
                self.v = timm.create_model('deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                # self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain) # 0.4.5
                self.v = timm.create_model('deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                # self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain) # 0.4.5
                self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.v.patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=192, in_chans=3)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches,
                                                                            self.original_embedding_dim).transpose(1,
                                                                                                                   2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :,
                                    int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(
                                        t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim),
                                                                    mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :,
                                    int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(
                                        f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError(
                    'currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                                   imagenet_pretrain=False, audioset_pretrain=False, model_size='base384',
                                   verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequencey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768,
                                                                                                              12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2): 50 - int(t_dim / 2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2): 6 - int(f_dim / 2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)

        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        try:
            x = x + self.v.pos_embed
        except Exception as ex:
            raise ex
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        # x = self.mlp_head(x)
        return x

class Patch_Embed(nn.Module):
    def __init__(self, kernel=(7,7), stride=(7,7), dim_in=3, dim_out=768, img_size = (224, 224), padding=(3, 3)):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(kernel)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_size = patch_size
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        # B C H W -> B HW C
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[-2:]

class MAST(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(self, label_dim=None, fstride=10, tstride=10, input_fdim=128, input_tdim=512, imagenet_pretrain=False,
                 audioset_pretrain=False, model_size='base', verbose=True, return_cls=False):

        super(MAST, self).__init__()

        if verbose == True:
            print('---------------MAST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),
                                                                                  str(audioset_pretrain)))

        timm.models.mvitv2.PatchEmbed = Patch_Embed
        self.mlp_head = None
        self.has_cls = False

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'large':
                self.v = timm.create_model('mvitv2_large', pretrained=imagenet_pretrain, cfg={'use_abs_pos':True})
            elif model_size == 'small':
                self.v = timm.create_model('mvitv2_small', pretrained=imagenet_pretrain)
            elif model_size == 'tiny':
                self.v = timm.create_model('mvitv2_tiny', pretrained=imagenet_pretrain)
            elif model_size == 'base':
                self.v = timm.create_model('mvitv2_base', pretrained=imagenet_pretrain)
            elif model_size == 'small_cls':
                self.v = timm.create_model('mvitv2_small_cls', pretrained=imagenet_pretrain)
                self.has_cls = True
            else:
                raise Exception('Model size must be one of tiny, small, base, large or small_cls.')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            # self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.original_embedding_dim = 96

            # optional to specify the last MLP layer for a specific class
            if label_dim is not None:
                self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches,
                                                                            self.original_embedding_dim).transpose(1,
                                                                                                                   2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :,
                                    int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(
                                        t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim),
                                                                    mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :,
                                    int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(
                                        f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                # self.v.patch_embed.num_patches + 1 if cls token
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError(
                    'currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = MAST(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                                    imagenet_pretrain=False, audioset_pretrain=False, model_size='base384',
                                    verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.original_embedding_dim = 96

            if label_dim is not None:
                self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768,
                                                                                                              12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2): 50 - int(t_dim / 2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2): 6 - int(f_dim / 2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x, patch_drop=0.0, return_cls=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x, x_shape = self.v.patch_embed(x)

        H, W = x_shape

        if patch_drop > 0:
            patch_keep = 1. - patch_drop
            T_H = int(np.floor((x.shape[1]) * patch_keep))
            perm = torch.randperm(x.shape[1])[:T_H]  # keep class token
            idx = torch.tensor(perm, dtype=perm.dtype, device=perm.device)
            x = x[:, idx, :]

        thw = [H, W]
        for blk in self.v.stages:
            # print(x.shape)
            x, thw = blk(x, thw)

        if self.has_cls and return_cls:
            x = self.v.norm(x)  # layer norm only if return_cls = False
            x = x[:, 0]
        else:
            x = x.mean(1)  # mean if no cls token

        # if self.mlp_head is not None:
        #     x = self.mlp_head(x)

        return x
# class TransformerEncoder(BaseSoundEncoder):
#     def __init__(self, sampling_rate=48000, fps=60, frame_skip=4, data_size=800):
#         super(TransformerEncoder, self).__init__(sampling_rate, fps, frame_skip)
#         self.window_size = int(self.sampling_rate * 0.025)
#         self.hop_size = int(self.sampling_rate * 0.01)
#         self.n_fft = int(self.sampling_rate * 0.025)
#         self.n_mels = 80
#         self.data_size = data_size
#
#         self.spectrogram = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self.sampling_rate,
#             n_mels=80,
#             n_fft=self.n_fft,
#             win_length=self.window_size,
#             hop_length=self.hop_size)
#         input_tdim, input_fdim = self.get_spectrogram_size()
#         self.ast_mdl = ASTModel(input_tdim=input_tdim, input_fdim=input_fdim, imagenet_pretrain=False)
#
#     def encode_single_channel(self, data):
#         x = torch.log(self.spectrogram(data) + 1e-5)
#         # x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#         x = self.ast_mdl(x)
#         return x
#
#     def get_spectrogram_size(self):
#         data = torch.randn((1, self.data_size))
#         output = self.spectrogram(data)
#         return output.shape[1], output.shape[2]


class TransformerEncoder(nn.Module):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4, data_size=800, num_mel_bins=128, model_size=None, transformer='ast'):
        super(TransformerEncoder, self).__init__()
        self.data_size = data_size
        self.sampling_rate = sampling_rate
        self.mel_bins = num_mel_bins
        input_tdim, input_fdim = self.get_size()
        if transformer == 'ast':
            self.ast_mdl = ASTModel(input_tdim=input_tdim, input_fdim=input_fdim, imagenet_pretrain=False,
                                model_size=model_size)
        elif transformer == 'mast':
            self.ast_mdl = MAST(input_tdim=input_tdim, input_fdim=input_fdim, imagenet_pretrain=False,
                                model_size=model_size)
        elif transformer == 'passt':
            self.ast_mdl = get_model('deit_tiny_distilled_patch16_224', input_tdim=input_tdim, input_fdim=input_fdim, pretrained=False)
        else:
            raise Exception('Transformer type is must be one of ast, mast or passt')

    def get_size(self):
        data = torch.randn((2, self.data_size))
        down_sampling = 48000 // self.sampling_rate
        fbank = torchaudio.compliance.kaldi.fbank(data[:, ::down_sampling], sample_frequency=self.sampling_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0,
                                                  frame_shift=10)
        return fbank.shape

    def forward(self, x):
        return self.ast_mdl(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FrameEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FrameEncoder, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, 128),
            Swish(),
            nn.Linear(128, latent_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)


class FrameEncoder1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FrameEncoder1, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)


class FrameEncoder2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FrameEncoder2, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)

class FrameEncoder3(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FrameEncoder3, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            Swish(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            Swish(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(0.7),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)


class FrameDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(FrameDecoder, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, 128),
            Swish(),
            nn.Linear(128, output_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)


class FrameDecoder1(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(FrameDecoder1, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)


class FrameDecoder2(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(FrameDecoder2, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)

class FrameDecoder3(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(FrameDecoder3, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            Swish(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            Swish(),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            Swish(),
        )

    def forward(self, x):
        return self.sequence(x)

if __name__ == '__main__':
    import numpy as np

    # torch.Tensor()
    np.random.seed(0)
    data = np.random.randn(1, 2, 16000)
    encoder = TransformerEncoder(data_size=16000, model_size='tiny224')

    # print(data)
    print(encoder(torch.Tensor(data)).shape)
    # print(encoder(torch.Tensor(data)))
