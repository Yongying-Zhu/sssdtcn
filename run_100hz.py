from config_100hz import config
from train_universal import train_model
from evaluate_universal import evaluate_model

train_model(config, "100Hz")
evaluate_model(config, "100Hz")
