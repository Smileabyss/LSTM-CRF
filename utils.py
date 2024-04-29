import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from args import args
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seqs, to_ix):
    # idxs = []
    # for seq in seqs:
    #     idx = [to_ix[w] for w in seq]
    #     idxs.append(idx)
    idxs = [to_ix[w] for w in seqs]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))