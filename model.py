import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import poptorch


class SimpleModel(nn.Module):
    def __init__(self, input_features, args):
        super().__init__()

        self.loss_type = args.loss_type
        if args.loss_type == "nll":
            output_size = 2
            self.loss = nn.CrossEntropyLoss()
        else:
            output_size = 1
            self.loss = nn.L1Loss()

        self.input_projection = nn.Linear(input_features, args.hidden_size)

        self.hidden = nn.ModuleList(nn.Linear(args.hidden_size, args.hidden_size)
                                    for _ in range(args.num_layers))

        self.batchnorm = nn.ModuleList(nn.BatchNorm1d(args.hidden_size)
                                       for _ in range(args.num_layers))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout_rate)

        self.output_projection = nn.Linear(args.hidden_size, output_size)

    def _num_correct_calculation(self, x, labels):
        predictions = torch.argmax(x, 1)
        diffs = torch.abs(predictions.int() - labels.int())
        correct = torch.ones(predictions.shape, dtype=torch.int) - diffs
        return predictions, torch.sum(correct)

    def _hidden_block(self, i, x):
        x = self.hidden[i](x)
        x = self.relu(x)
        x = self.batchnorm[i](x)
        return self.dropout(x)

    def forward(self, input_tensor, label_tensor):
        x = self.input_projection(input_tensor)
        x = self.relu(x)
        # for i in range(len(self.hidden)):
        x = self._hidden_block(0, x)
        x = self.output_projection(x)

        correct = self._num_correct_calculation(x, label_tensor)

        if self.loss_type != "nll":
            x = torch.sigmoid(x)

        loss = self.loss(x, label_tensor.flatten())
        return (loss, ) + correct # loss, predictions, num_correct
