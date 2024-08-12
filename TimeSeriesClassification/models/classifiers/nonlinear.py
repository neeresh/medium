from torch import nn


class NonlinearClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, dropout=0.2):
        super(NonlinearClassifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, embedding_dim)
        self.batchnorm1 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embedding_dim, output_dim),
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x
