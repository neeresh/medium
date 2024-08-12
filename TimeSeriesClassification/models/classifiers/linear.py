from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LinearClassifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.dense(x))
