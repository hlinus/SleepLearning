import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def kl_div(p, q):
    return (p * (p.clamp(min=1e-7).log() - q.clamp(min=1e-7).log())).sum()


class GrangerLoss(nn.Module):
    def __init__(self, weights, alpha):
        super().__init__()
        self.alpha = alpha
        # weighted cross encropy loss per sample in batch
        self.xeloss = CrossEntropyLoss(weight=weights, reduce=False)

    def forward(self, y_, y_true):
        yaux_i, yaux_c, y_att, attention = y_
        # List[bs x 1]
        eps_X_i = [self.xeloss(yaux_i_, y_true) for yaux_i_ in yaux_i]
        # [nchannels x bs]
        eps_X_i = torch.stack(eps_X_i, 0)
        # [bs]
        eps_X = self.xeloss(yaux_c, y_true)
        # [nchannels x bs]
        delta_eps_X_i = eps_X_i - eps_X
        # [nchannels x bs]
        omega = delta_eps_X_i / torch.sum(delta_eps_X_i)
        # List[1]
        l_mge = [kl_div(omega_i, a) for (omega_i, a) in zip(omega.permute(1,
                                                                          0),
                                                            attention)]
        # [1]
        l_mge = torch.mean(torch.stack(l_mge,0))
        l_main = self.xeloss(y_att, y_true).mean()

        return (1-self.alpha)*l_main + self.alpha*l_mge


if __name__ == '__main__':
    class_weights = torch.ones((5,1))
    bs = 10
    nchannels = 7
    nclasses = 5
    attention = torch.randn(bs, nchannels, requires_grad=True)
    attention = F.softmax(attention, dim=1)
    yaux_i = [torch.randn(bs, nclasses, requires_grad=True) for _ in
              range(nchannels)]
    yaux_c = torch.randn(bs, nclasses, requires_grad=True)
    y_att = torch.randn(bs, nclasses, requires_grad=True)

    y_true = torch.empty(bs, dtype=torch.long).random_(5)
    gl = GrangerLoss(class_weights, 0.1)
    y = (yaux_i, yaux_c, y_att, attention)
    print(gl(y, y_true))
