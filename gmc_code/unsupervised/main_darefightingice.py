import os
import torch
import sacred
import gmc_code.unsupervised.ingredients.exp_ingredients as sacred_exp
import gmc_code.unsupervised.ingredients.machine_ingredients as sacred_machine
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.unsupervised.modules.trainers.model_trainer import ModelLearner
from gmc_code.unsupervised.modules.sacred_loggers import SacredLogger
from gmc_code.unsupervised.modules.callbacks import OnEndModelTrainingMHD, OnEndDownTrainingMHD
from gmc_code.unsupervised.utils.general_utils import (
    setup_dca_evaluation_trainer,
    setup_model,
    setup_data_module,
    load_model,
    setup_downstream_classifier,
    setup_downstream_classifier_trainer,
    load_down_model,
    setup_downstream_evaluator,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "DareFightingICE_transformers_experiments",
    ingredients=[sacred_machine.machine_ingredient, sacred_exp.exp_ingredient],
)