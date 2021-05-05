import torch
import torch.nn as nn
import torch.nn.functional as F

class WordCNN(nn.Module):

    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(WordCNN, self).__init__()

        V = vocab_size + 1
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = []
        for i in args.kernel_sizes.split(","):
            Ks.append(int(i))

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):

        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit