from gmc_code.unsupervised.architectures.models.gmc import MhdGMC, ICEGMC
from gmc_code.unsupervised.architectures.downstream.classifiers import ClassifierMNIST
from gmc_code.unsupervised.modules.trainers.classification_trainer import ClassifierLearner
from gmc_code.unsupervised.modules.trainers.dca_evaluation_trainer import DCAEvaluator
from gmc_code.unsupervised.data_modules.class_dataset import *
from gmc_code.unsupervised.architectures.models.transforms import *
from torchvision import transforms as T
import torch


def setup_model(scenario, model, model_config, scenario_config, data_module=None):
    if model == "gmc":
        if scenario == "mhd":
            return MhdGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
            )
        #
        # else:
        #     raise ValueError(
        #         "[Model Setup] Selected scenario not yet implemented for GMC model: "
        #         + str(scenario)
        #     )
        # elif model == 'transformers':
        if scenario == 'transformers':
            return ICEGMC(
                name=model_config['model'],
                common_dim=model_config['common_dim'],
                latent_dim=model_config['latent_dim'],
                loss_type=model_config['loss_type'],
                frame_size=model_config['frame_size'][scenario_config['predict_type']],
                sound_length=model_config['sound_length'],
                model_size=model_config['model_size'],
                sampling_rate=scenario_config['sampling_rate'],
                num_mel_bins=model_config['num_mel_bins'],
                frame_encoder=model_config['frame_encoder'],
                weight_gmc=model_config['weight_gmc'],
                weight_reconstruct=model_config['weight_reconstruct'],
                active_mod=model_config['active_mod'],
                frame_model_pretrain=model_config['pretrain_frame'],
                sound_model_pretrain=model_config['pretrain_sound'],
                transformer_type=model_config['transformer_type']
            )
        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for GMC model: "
                + str(scenario)
            )
    else:
        raise ValueError(
            "[Model Setup] Selected model not yet implemented: " + str(model)
        )


def setup_data_module(scenario, experiment_config, scenario_config, train_config, model_config):
    print('dataset in use:', [scenario_config['dataset']])
    if scenario == "mhd":
        if experiment_config["stage"] == "evaluate_dca":
            return DCADataModule(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                data_config=train_config,
            )
        else:
            return ClassificationDataModule(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                data_config=train_config,
            )
    elif scenario == 'transformers':
        transforms = {
            'all': T.Compose([ToTensor()]),
            'Frame': T.Compose([Mask(scenario_config['frame_mask'][scenario_config['predict_type']])]),
            # 'Audio': T.Compose([Transpose(), Cutoff()])
            'Audio': T.Compose([Cutoff2()])
        }
        # return ICEDataModule(root_dir=scenario_config['data_dir'], transforms=transforms,
        #                      batch_size=experiment_config['model_train_config']['batch_size'],
        #                      frame_col='frameDataNonDelay')
        return ICEDataModule2(train_dir=scenario_config['data_dir'][scenario_config['dataset']][0],
                              test_dir=scenario_config['data_dir'][scenario_config['dataset']][1],
                              batch_size=experiment_config['model_train_config']['batch_size'],
                              frame_mask=scenario_config['frame_mask'][scenario_config['predict_type']],
                              sampling_rate=scenario_config['sampling_rate'],
                              data_config=train_config,
                              sound_length=model_config['sound_length'],
                              num_mel_bins=model_config['num_mel_bins'],
                              transition=model_config['transition'],
                              transition_mask=scenario_config['transition_mask'],
                              active_mod=model_config['active_mod'])
    else:
        raise ValueError(
            "[Data Module Setup] Selected Module not yet implemented: " + str(scenario)
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )


def setup_downstream_classifier(scenario, model_config):
    if scenario == 'mhd':  # Classification
        return ClassifierMNIST(latent_dim=model_config["latent_dim"])
    else:
        raise ValueError(
            "[Down Model Setup] Selected scenario not yet implemented: " + str(scenario)
        )


def setup_downstream_classifier_trainer(scenario, rep_model, down_model, train_config):
    if scenario == 'mhd':  # Classification
        return ClassifierLearner(
            model=rep_model, scenario=scenario, classifier=down_model, train_config=train_config,
        )

    else:
        raise ValueError(
            "[Down Classifier Trainer Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


def setup_downstream_evaluator(
        scenario, rep_model, down_model, train_config, modalities
):
    if scenario == 'mhd':  # Classification
        return ClassifierLearner(
            model=rep_model,
            scenario=scenario,
            classifier=down_model,
            train_config=train_config,
            modalities=modalities,
        )
    else:
        raise ValueError(
            "[Down Evaluator Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


"""

Loading functions

"""


def load_model(sacred_config, model_file):
    model = setup_model(
        scenario=sacred_config["experiment"]["scenario"],
        model=sacred_config["experiment"]["model"],
        model_config=sacred_config["experiment"]["model_config"],
        scenario_config=sacred_config["experiment"]["scenario_config"],
    )

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()

    return model


def load_down_model(sacred_config, down_model_file):
    down_model = setup_downstream_classifier(
        scenario=sacred_config["experiment"]["scenario"],
        model_config=sacred_config["experiment"]["model_config"],
    )

    checkpoint = torch.load(down_model_file)
    down_model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    down_model.freeze()
    down_model.eval()

    return down_model


"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


