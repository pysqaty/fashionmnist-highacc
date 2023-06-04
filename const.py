""" Main script parameters """
BATCH_SIZE = 32
NEPOCHS = 300
PATIENCE = 250
NWORKERS = 1
SEED = 1
# set to False if you want to only evaluate model on both: training and test sets
DO_TRAINING = False
# set path to None if you want to use clean model
MODEL_PATH = "./models/model-0.9505.pth.tar"
