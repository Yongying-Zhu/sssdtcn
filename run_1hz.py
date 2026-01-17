from config_1hz import config
from train_universal import train_model
from evaluate_universal import evaluate_model

train_model(config, "1Hz")
evaluate_model(config, "1Hz")
