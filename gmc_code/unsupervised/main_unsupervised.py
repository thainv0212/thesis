import os
import torch
import sacred
import gmc_code.unsupervised.ingredients.exp_ingredients as sacred_exp
import gmc_code.unsupervised.ingredients.machine_ingredients as sacred_machine
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from gmc_code.unsupervised.modules.trainers.model_trainer import ModelLearner
from gmc_code.unsupervised.modules.sacred_loggers import SacredLogger
from gmc_code.unsupervised.modules.callbacks import OnEndModelTrainingMHD, OnEndDownTrainingMHD, OnEndDownTrainingICE, \
    OnEndModelTrainingICE, KLWeightScheduler
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
    "GMC_unsupervised_experiments",
    ingredients=[sacred_machine.machine_ingredient, sacred_exp.exp_ingredient],
)


@ex.capture
def log_dir_path(folder, _config, _run):
    model_type = str(_config["experiment"]["model"])
    exp_name = str(_config["experiment"]["scenario"])

    return os.path.join(
        _config["machine"]["m_path"],
        "evaluation/",
        model_type + "_" + exp_name,
        f'log_{_config["experiment"]["seed"]}',
        folder,
    )


@ex.capture
def trained_model_dir_path(file, _config, _run):
    return os.path.join(
        _config["machine"]["m_path"],
        "trained_models/",
        file
    )


@ex.capture
def load_hyperparameters(_config, _run):
    exp_cfg = _config["experiment"]
    scenario_cfg = _config["experiment"]["scenario_config"]
    model_cfg = _config["experiment"]["model_config"]

    return exp_cfg, scenario_cfg, model_cfg


@ex.capture
def train_model(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    model_train_cfg = _config["experiment"]["model_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Init model
    model = setup_model(
        model=exp_cfg["model"],
        scenario=exp_cfg["scenario"],
        scenario_config=scenario_cfg,
        model_config=model_cfg)

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
        model_config=model_cfg
    )

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg)

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)
    tensorboard_logger = TensorBoardLogger(
        f"trained_models/evaluation/gmc_transformers/{model_cfg['transformer_type']}/{scenario_cfg['name']}_{scenario_cfg['predict_type']}_{scenario_cfg['dataset']}_modality{model_cfg['active_mod']}_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_{model_cfg['common_dim']}_{model_cfg['latent_dim']}_{model_cfg['sound_length']}_{model_cfg['model_size']}_{scenario_cfg['sampling_rate']}_{model_cfg['num_mel_bins']}-FrameEncoder{model_cfg['frame_encoder']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-transition-{model_cfg['transition']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-{model_cfg['interval']}-{model_cfg['weight_decay']}-{scenario_cfg['transition_mask']}",
        name=exp_cfg["scenario"])

    # Train
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{scenario_cfg['name']}_{model_cfg['transformer_type']}_{scenario_cfg['predict_type']}_{scenario_cfg['dataset']}_modality{model_cfg['active_mod']}_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_{model_cfg['common_dim']}_{model_cfg['latent_dim']}_{model_cfg['sound_length']}_{model_cfg['model_size']}_{scenario_cfg['sampling_rate']}_{model_cfg['num_mel_bins']}-FrameEncoder{model_cfg['frame_encoder']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-transition-{model_cfg['transition']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-{model_cfg['interval']}-{model_cfg['weight_decay']}-{scenario_cfg['transition_mask']}"
                 + "{epoch:02d}",
        monitor="total_loss_val",
        every_n_epochs=model_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"{scenario_cfg['name']}_{model_cfg['transformer_type']}_{scenario_cfg['predict_type']}_{scenario_cfg['dataset']}_modality{model_cfg['active_mod']}_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_{model_cfg['common_dim']}_{model_cfg['latent_dim']}_{model_cfg['sound_length']}_{model_cfg['model_size']}_{scenario_cfg['sampling_rate']}_{model_cfg['num_mel_bins']}-FrameEncoder{model_cfg['frame_encoder']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-transition-{model_cfg['transition']}-{model_cfg['weight_gmc']}-{model_cfg['weight_reconstruct']}-{model_cfg['interval']}-{model_cfg['weight_decay']}-{scenario_cfg['transition_mask']}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    weight_scheduler = None
    if exp_cfg["scenario"] == "mhd":
        end_callback = OnEndModelTrainingMHD()
    elif exp_cfg["scenario"] == "transformers":
        end_callback = OnEndModelTrainingICE()
        if model_cfg['enable_weight_decay'] == 1:
            weight_scheduler = KLWeightScheduler(interval=model_cfg['interval'], decay=model_cfg['weight_decay'])
    else:
        raise ValueError("Error")
    # TEST_limit_train_batches = 0.01
    if weight_scheduler is not None:
        callbacks = [checkpoint_callback, end_callback, weight_scheduler]
    else:
        callbacks = [checkpoint_callback, end_callback]
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=model_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=LoggerCollection([sacred_logger, tensorboard_logger]),
        callbacks=callbacks)

    # Train
    trainer.fit(model_trainer, data_module)


@ex.capture
def dca_eval_model(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, _ = load_hyperparameters()
    dca_eval_cfg = _config["experiment"]["dca_evaluation_config"]

    # Set the seeds
    seed_everything(dca_eval_cfg["random_seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model = load_model(sacred_config=_config, model_file=model_file)

    # Init Trainer
    dca_trainer = setup_dca_evaluation_trainer(
        model=model,
        machine_path=_config["machine"]["m_path"],
        scenario=exp_cfg["scenario"],
        config=dca_eval_cfg,
    )

    # Init Data Module
    dca_data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=dca_eval_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("results_dca_evaluation"),
        logger=sacred_logger,
    )

    trainer.test(dca_trainer, dca_data_module)
    return


@ex.capture
def train_downstream_classifer(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["experiment"]["down_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Init downstream model
    down_model = setup_downstream_classifier(
        scenario=exp_cfg["scenario"], model_config=model_cfg
    )

    # Init Trainer
    down_trainer = setup_downstream_classifier_trainer(
        scenario=exp_cfg["scenario"],
        rep_model=rep_model,
        down_model=down_model,
        train_config=down_train_cfg,
    )
    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Callbacks
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"down_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
                 + "{epoch:02d}",
        monitor="total_loss_val",
        every_n_epochs=down_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"down_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    if exp_cfg["scenario"] == "mhd":
        end_callback = OnEndDownTrainingMHD()
    elif exp_cfg["scenario"] == "transformers":
        end_callback = OnEndDownTrainingICE()
    else:
        raise ValueError("Error")

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=down_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        callbacks=[checkpoint_callback, end_callback])

    trainer.fit(down_trainer, data_module)


@ex.capture
def eval_downstream_classifier(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["experiment"]["down_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Load downstream model
    down_model_file = trained_model_dir_path("down_" + exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    down_model = load_down_model(sacred_config=_config, down_model_file=down_model_file)

    # Init Trainer
    down_trainer = setup_downstream_evaluator(
        scenario=exp_cfg["scenario"],
        rep_model=rep_model,
        down_model=down_model,
        train_config=down_train_cfg,
        modalities=_config["experiment"]["evaluation_mods"],
    )

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=down_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        log_every_n_steps=10
    )

    trainer.test(down_trainer, data_module)


@ex.main
def main(_config, _run):
    # Run experiment
    if _config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_model()

    elif _config["experiment"]["stage"] == "evaluate_dca":
        os.makedirs(log_dir_path("results_dca_evaluation"), exist_ok=True)
        dca_eval_model()

    elif _config["experiment"]["stage"] == "train_downstream_classfier":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_downstream_classifer()

    elif _config["experiment"]["stage"] == "evaluate_downstream_classifier":
        os.makedirs(log_dir_path("results_down"), exist_ok=True)
        eval_downstream_classifier()

    else:
        raise ValueError(
            "[Unsupervised Experiment] Incorrect stage of pipeline selected: " + str(_config["experiment"]["stage"])
        )


if __name__ == "__main__":
    ex.run_commandline()
