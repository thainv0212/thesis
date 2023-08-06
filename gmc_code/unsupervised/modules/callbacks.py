import os
import torch
from pytorch_lightning.callbacks import Callback


class OnEndModelTrainingMHD(Callback):
    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):
        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir, f"{pl_module.model.name}_mhd_model.pth.tar"
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the MHD dataset saved to {trainer.default_root_dir}"
        )


class KLWeightScheduler(Callback):

    def __init__(self, interval=1, decay=0.1):
        self.counter = 0
        self.interval = interval
        self.decay = decay

    # def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
    #     self.counter = 0
    #     self.interval = 10
    #     self.decay = 0.1
    #     pass

    def clamp(self, n, smallest, largest):
        return max(smallest, min(n, largest))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.counter += 1
        if self.counter % self.interval == 0:
            pl_module.model.weight_gmc += self.decay
            pl_module.model.weight_gmc = self.clamp(pl_module.model.weight_gmc, 0, 1)
            pl_module.model.weight_reconstruction -= self.decay
            pl_module.model.weight_reconstruction = self.clamp(pl_module.model.weight_reconstruction, 0, 1)
            print(f'Perform weight adjustment, gmc: {pl_module.model.weight_gmc}, {pl_module.model.weight_reconstruction}')
        return


class OnEndDownTrainingMHD(Callback):
    def on_init_end(self, trainer):
        print(
            f"Initialised Trainer for MHD downstream with {trainer.default_root_dir}"
        )

    def on_train_end(self, trainer, pl_module):
        torch.save(
            {"state_dict": pl_module.classifier.state_dict()},
            os.path.join(
                trainer.default_root_dir,
                f"down_{pl_module.model.name}_mhd_model.pth.tar",
            ),
        )

        print(
            f"Downstream classifier for representation model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the MHD dataset saved to {trainer.default_root_dir}"
        )


class OnEndModelTrainingICE(Callback):
    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):
        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir, f"{pl_module.model.name}_ice_model.pth.tar"
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the ICE dataset saved to {trainer.default_root_dir}"
        )


class OnEndDownTrainingICE(Callback):
    def on_init_end(self, trainer):
        print(
            f"Initialised Trainer for ICE downstream with {trainer.default_root_dir}"
        )

    def on_train_end(self, trainer, pl_module):
        torch.save(
            {"state_dict": pl_module.classifier.state_dict()},
            os.path.join(
                trainer.default_root_dir,
                f"down_{pl_module.model.name}_ice_model.pth.tar",
            ),
        )

        print(
            f"Downstream classifier for representation model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the ICE dataset saved to {trainer.default_root_dir}"
        )
