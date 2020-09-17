import torch
import torch.utils.data
import torch.optim
import poptorch
from tqdm import tqdm

from model import SimpleModel
import dataset
import utils
import wandb


def valid(args, inference_model, data):

    num_batches = len(data) // args.batch_size

    opts = poptorch.Options()
    opts.deviceIterations(args.batches_per_step)
    data_loader = poptorch.DataLoader(opts, data, args.batch_size, True)

    total_correct = 0
    total_processed = 0

    utils.logger.info("Starting validation")
    with torch.no_grad():
        for data, labels in tqdm(data_loader, total=num_batches):
            loss, predictions, num_correct = inference_model(data, labels)

            total_correct += int(num_correct)
            total_processed += len(predictions)

    accuracy = 100 * total_correct / total_processed
    utils.logger.info(f"Validation accuracy: {accuracy}")
    return accuracy


def train(args):
    training_data, validation_data = dataset.load(args.dataset, args.train_dataset_proportion)

    if args.batch_size is None:
        args.batch_size = len(training_data)

    # We're using 3 features in the dataset, so this is fixed
    input_features = 3

    opts = poptorch.Options()
    opts.deviceIterations(args.batches_per_step)

    data_loader = poptorch.DataLoader(opts, training_data, args.batch_size, True)

    model = SimpleModel(input_features, args)

    training_model = poptorch.trainingModel(model,
                                            opts,
                                            optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
    inference_model = poptorch.inferenceModel(model)

    for i in range(args.num_epochs):
        for step, batch in enumerate(data_loader):
            # Lazily discard items that don't fit into a full batch
            if len(batch[0]) != args.batch_size:
                continue

            loss, predictions, num_correct = training_model(batch[0], batch[1])
            accuracy = 100 * num_correct.detach().numpy()[0] / len(predictions)

            log_data = {
                "epoch": i,
                "loss": loss,
                "train_acc": accuracy
            }

            if i % args.epochs_per_log == 0:
                utils.logger.info(f"Epoch {i} | Loss: {loss.numpy()[0]:.4f} | Accuracy: {accuracy:.2f}")

            if i % args.epochs_per_validation == 0:
                training_model.copyWeightsToHost()
                log_data["valid_acc"] = valid(args, inference_model, validation_data)

            if args.wandb:
                wandb.log(log_data)

    utils.logger.info("Training Complete")

    utils.logger.info("Final Validation...")
    training_model.copyWeightsToHost()
    valid(args, inference_model, validation_data)


if __name__ == "__main__":
    args = utils.parse_args()

    if args.wandb:
        wandb.init(project="boats")
        wandb.config.update(args)

    if args.mode == "train":
        train(args)
