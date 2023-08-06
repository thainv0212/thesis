import torch
from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.models.gmc_networks import *


class GMC(LightningModule):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce"):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None

    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:
            return self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def forward(self, x):

        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                    torch.ones_like(sim_matrix_joint_mod)
                    - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def infonce_with_joints_as_negatives(
            self, batch_representations, temperature, batch_size
    ):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / temperature
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
                torch.ones_like(sim_matrix_joints)
                - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(
            batch_size, -1
        )

        # compute loss - for each pair joints-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return tqdm_dict


class MhdGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce"):
        super(MhdGMC, self).__init__(name, common_dim, latent_dim, loss_type)

        self.image_processor = MHDImageProcessor(common_dim=common_dim)
        self.sound_processor = MHDSoundProcessor(common_dim=common_dim)
        self.trajectory_processor = MHDTrajectoryProcessor(common_dim=common_dim)
        self.label_processor = MHDLabelProcessor(common_dim=common_dim)
        self.joint_processor = MHDJointProcessor(common_dim=common_dim)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.trajectory_processor,
            self.label_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type

        self.encoder = MHDCommonEncoder(common_dim=common_dim, latent_dim=latent_dim)


class GMCTransformers(LightningModule):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", weight_gmc=None, weight_reconstruct=None):
        super(GMCTransformers, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.sound_processor = None
        self.frame_processor = None
        self.joint_processor = None
        self.processors = [
            self.sound_processor,
            self.frame_processor,
            self.joint_processor,
        ]

        self.encoder = ICECommonEncoder(common_dim=common_dim, latent_dim=latent_dim)
        self.frame_decoder = None
        self.weight_gmc = weight_gmc
        self.weight_reconstruction = weight_reconstruct

    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:
            return self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def construct_frame(self, x):
        x = [x, None]
        latent = self.encode(x)
        return self.frame_decoder(latent)

    def forward(self, x):
        batch_representations = []
        # Forward pass through the modality specific encoders
        if len(self.processors) > 1:
            for processor_idx in range(len(self.processors) - 1):
                mod_representations = self.encoder(
                    self.processors[processor_idx](x[processor_idx])
                )
                batch_representations.append(mod_representations)
            # Forward pass through the joint encoder
            joint_representation = self.encoder(self.processors[-1](x))
            batch_representations.append(joint_representation)
            # Reconstruct frame data
            reconstructed_frame = self.frame_decoder(joint_representation)
            batch_representations.append(reconstructed_frame)
        else:  # If there is only one modality available
            mod_representation = self.encoder(self.processors[0](x[0]))
            batch_representations.append(mod_representation)
            # Reconstruct frame data
            reconstructed_frame = self.frame_decoder(mod_representation)
            batch_representations.append(reconstructed_frame)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                    torch.ones_like(sim_matrix_joint_mod)
                    - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss_gmc": loss}
        return loss, tqdm_dict

    def infonce_with_joints_as_negatives(
            self, batch_representations, temperature, batch_size
    ):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / temperature
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
                torch.ones_like(sim_matrix_joints)
                - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(
            batch_size, -1
        )

        # compute loss - for each pair joints-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss_gmc": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]
        if torch.isnan(data[0]).sum() > 0:
            print("error")
        # Forward pass through the encoders
        batch_representations = self.forward(data[:-1])

        # Compute contrastive loss
        # TODO add reconstruction loss
        if len(batch_representations) > 2:
            if self.loss_type == "infonce_with_joints_as_negatives":
                loss_gmc, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations[:-1], temperature,
                                                                            batch_size)
            else:
                loss_gmc, tqdm_dict = self.infonce(batch_representations[:-1], temperature, batch_size)
        else:
            loss_gmc = 0
            tqdm_dict = {'loss_gmc': torch.tensor(0, dtype=torch.float32)}

        frames = data[-1]
        reconstructed_frames = batch_representations[-1]
        loss_reconstruction = F.mse_loss(reconstructed_frames, frames)
        loss = self.weight_gmc * loss_gmc + self.weight_reconstruction * loss_reconstruction
        # if torch.isnan(loss).cpu().detach().numpy() == True:
        #     print('exception')
        tqdm_dict['loss_reconstruction_train'] = loss_reconstruction
        tqdm_dict['total_loss_train'] = loss  # tqdm_dict['loss_gmc'] + loss_reconstruction
        tqdm_dict['loss_gmc_train'] = tqdm_dict['loss_gmc']
        del tqdm_dict['loss_gmc']

        tqdm_dict['weight_gmc'] = torch.tensor(self.weight_gmc, dtype=torch.float32)
        tqdm_dict['weight_reconstruction'] = torch.tensor(self.weight_reconstruction, dtype=torch.float32)
        # self.log_dict(tqdm_dict, batch_size=batch_size, sync_dist=True)
        return loss, tqdm_dict

    def reconstruction_loss(self, legits, labels):
        return F.mse_loss(legits, labels)

    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data[:-1])
        # Compute contrastive loss
        # TODO add frame reconstruction loss
        if len(batch_representations) > 2:
            if self.loss_type == "infonce_with_joints_as_negatives":
                loss_gmc, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations[:-1], temperature,
                                                                            batch_size)
            else:
                loss_gmc, tqdm_dict = self.infonce(batch_representations[:-1], temperature, batch_size)
        else:
            loss_gmc = 0
            tqdm_dict = {'loss_gmc': torch.tensor(0, dtype=torch.float32)}
        frames = data[-1]
        reconstructed_frames = batch_representations[-1]
        loss_reconstruction = F.mse_loss(reconstructed_frames, frames)
        loss = self.weight_gmc * loss_gmc + self.weight_reconstruction * loss_reconstruction
        tqdm_dict['loss_reconstruction_val'] = loss_reconstruction
        tqdm_dict['total_loss_val'] = loss  # tqdm_dict['loss_gmc'] + loss_reconstruction
        tqdm_dict['loss_gmc_val'] = tqdm_dict['loss_gmc']
        del tqdm_dict['loss_gmc']
        # self.log_dict(tqdm_dict, batch_size=batch_size, sync_dist=True)
        return tqdm_dict


class ICEGMC(GMCTransformers):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", frame_size=None, sound_length=None,
                 num_mel_bins=128, sampling_rate=48000, model_size=None, frame_encoder=None, weight_gmc=None,
                 weight_reconstruct=None, active_mod=None, sound_model_pretrain=None, frame_model_pretrain=None,
                 transformer_type=None):
        super(ICEGMC, self).__init__(name, common_dim, latent_dim, loss_type, weight_gmc=weight_gmc,
                                     weight_reconstruct=weight_reconstruct)

        if active_mod is None:
            self.frame_processor = ICEFrameProcessor(frame_size, common_dim, encoder_type=frame_encoder,
                                                     frame_model_pretrain=frame_model_pretrain)
            self.sound_processor = ICESoundProcessor(sound_data_length=sound_length, common_dim=common_dim,
                                                     num_mel_bins=num_mel_bins, sampling_rate=sampling_rate,
                                                     model_size=model_size, sound_model_pretrain=sound_model_pretrain,
                                                     transformer_type=transformer_type)
            self.joint_processor = ICEJointProcessor(common_dim, sound_length, frame_size, num_mel_bins,
                                                     sampling_rate=sampling_rate, model_size=model_size,
                                                     frame_encoder_type=frame_encoder,
                                                     sound_model_pretrain=sound_model_pretrain,
                                                     frame_model_pretrain=frame_model_pretrain,
                                                     transformer_type=transformer_type)
            self.processors = [
                self.sound_processor,
                self.frame_processor,
                self.joint_processor,
            ]
        elif active_mod == 'sound':
            self.sound_processor = ICESoundProcessor(sound_data_length=sound_length, common_dim=common_dim,
                                                     num_mel_bins=num_mel_bins, sampling_rate=sampling_rate,
                                                     model_size=model_size, sound_model_pretrain=sound_model_pretrain,
                                                     transformer_type=transformer_type)
            self.processors = [self.sound_processor]
        elif active_mod == 'frame':
            self.frame_processor = ICEFrameProcessor(frame_size, common_dim, encoder_type=frame_encoder,
                                                     frame_model_pretrain=frame_model_pretrain)
            self.processors = [self.frame_processor]
        elif active_mod == 'joint':
            self.joint_processor = ICEJointProcessor(common_dim, sound_length, frame_size, num_mel_bins,
                                                     sampling_rate=sampling_rate, model_size=model_size,
                                                     frame_encoder_type=frame_encoder,
                                                     sound_model_pretrain=sound_model_pretrain,
                                                     frame_model_pretrain=frame_model_pretrain,
                                                     transformer_type=transformer_type)
            self.processors = [self.joint_processor]
        else:
            raise Exception("Active modality is not valid")
        self.loss_type = loss_type
        self.encoder = ICECommonEncoder(common_dim=common_dim, latent_dim=latent_dim)
        self.frame_decoder = ICEFrameReconstructor(latent_dim=latent_dim, output_dim=frame_size,
                                                   decoder_type=frame_encoder)

    def load_checkpoint(self, model_file):
        checkpoint = torch.load(model_file)['state_dict']
        for key in list(checkpoint.keys()):
            if 'model.' in key:
                checkpoint[key.replace('model.', '')] = checkpoint[key]
                del checkpoint[key]
        self.load_state_dict(checkpoint)
        # self.eval()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    model = ICEGMC(name='model', common_dim=20, latent_dim=15, frame_size=30, sound_length=24000, num_mel_bins=128,
                   sampling_rate=48000, transformer_type='ast',
                   model_size='tiny224', frame_encoder=0).to(device)
    # model.load_checkpoint(
    #     '/home/thai/gmc_transformers_None_20_15_24000_tiny224-epoch=00.pth')
    # model.float()

    # model.eval()
    # total_params = sum(p.numel() for p in model.parameters())
    # print('total size', total_params)

    # import time
    #
    audio = torch.randn((1, 48, 128)).to(device)

    # frame = torch.randn((2, 30)).to(device)
    data = torch.randn((2, 16000)).to(device)
    # start = time.time()
    mel = torchaudio.compliance.kaldi.fbank(data, htk_compat=True, sample_frequency=16000,
                                            use_energy=False, window_type='hanning', num_mel_bins=32, dither=0.0,
                                            frame_shift=10)
    from torchsummary import summary
    # print(summary(model, ([1] + list(mel.shape))))
    # model.construct_frame((audio, None))
    #
    # for _ in range(1000):
    #     torchaudio.compliance.kaldi.fbank(data, htk_compat=True, sample_frequency=16000,
    #                                       use_energy=False, window_type='hanning', num_mel_bins=16, dither=0.0,
    #                                       frame_shift=10)
    #     model.construct_frame((audio, None))
    # end = time.time()
    # print('time used', (end - start) * 1000 / 1000)
    # data = torch.randn((2, 24000)).to(device)
    # print(torchaudio.compliance.kaldi.fbank(data, htk_compat=True, sample_frequency=16000,
    #                                         use_energy=False, window_type='hanning', num_mel_bins=16, dither=0.0,
    #                                         frame_shift=10).shape)
