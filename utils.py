import logging
import argparse

logger = logging.getLogger("boats")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mode", default="train", choices=("train", "test"), type=str)
    parser.add_argument("--batches_per_step", default=1, type=int)
    parser.add_argument("--epochs_per_validation", default=5, type=int)
    parser.add_argument("--epochs_per_log", default=1, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--train_dataset_proportion", default=0.7, type=float)

    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--hidden_size", type=int, default=20)
    model_group.add_argument("--num_layers", type=int, default=1)
    model_group.add_argument("--loss_type", default="nll", choices=("nll", "l1"), type=str)
    model_group.add_argument("--batch_size", default=None, type=int)

    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--dropout_rate", type=float, default=0.1)
    train_group.add_argument("--num_epochs", type=int, default=100)
    train_group.add_argument("--learning_rate", type=float, default=0.01)

    return parser.parse_args()
