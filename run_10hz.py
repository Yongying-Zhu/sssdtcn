from config_10hz import config
from train_universal import train_model
from evaluate_universal import evaluate_model

train_model(config, "10Hz")
evaluate_model(config, "10Hz")
