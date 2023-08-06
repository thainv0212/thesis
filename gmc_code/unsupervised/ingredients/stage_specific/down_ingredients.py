import sacred

###################################
#       Downstream  Train         #
###################################

down_train_ingredient = sacred.Ingredient("down_train")


@down_train_ingredient.named_config
def mhd():
    # Dataset parameters
    batch_size = 64
    num_workers = 112

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None

@down_train_ingredient.named_config
def ice():
    # Dataset parameters
    batch_size = 64
    num_workers = 16

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-7
    snapshot = 25
    checkpoint = None