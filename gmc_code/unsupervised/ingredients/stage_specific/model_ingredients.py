import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")


@model_ingredient.config
def gmc_mhd():
    model = "gmc"
    common_dim = 64
    latent_dim = 64
    loss_type = "infonce"  # "joints_as_negatives"

@model_ingredient.config
def gmc_ice():
    model = "gmc"
    common_dim = 20
    latent_dim = 15
    loss_type = "infonce"  # "joints_as_negatives"
    frame_size = {
        'normal': 30,
        'no_y_speed': 26
    }
    sound_length = 16000
    model_size = 'tiny224'
    num_mel_bins = 16
    frame_encoder = 1
    weight_gmc = 0.00
    weight_reconstruct = 1.00
    enable_weight_decay = 0
    transition = 1
    interval = 5
    weight_decay = 0.01
    active_mod = None
    pretrain_sound = None
    pretrain_frame = None
    transformer_type = None

##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.1

@model_train_ingredient.named_config
def gmc_ice_train():
    # Dataset parameters
    # data_dir = "../../sample/"
    batch_size = 64
    num_workers = 112

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-6
    snapshot = 1
    checkpoint = None
    temperature = 0.1