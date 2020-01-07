import torch
import torch.nn as nn

def f_score(predicted, actual, beta = 1, eps = 1e-7, threshold = None, activation = 'sigmoid'):

    if activation is None or activation == 'none':
        activation_function = lambda x: x
    elif activation == 'sigmoid':
        activation_function = torch.nn.Sigmoid()
    elif activation == 'softmax':
        activation_function = torch.nn.Softmax()
    else:
        raise NotImplementedError("Activation function is implemented for only sigmoid and softmax")

    predicted = activation_function(predicted)

    if threshold is not None:
        predicted = (predicted > threshold).float()

    tp = torch.sum(actual * predicted)
    fp = torch.sum(predicted) - tp
    fn = torch.sum(actual) - tp

    score = ((1 + beta ** 2) * tp * eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'
    def __init__(self, eps = 1e-7, activation = 'sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, predicted, actual):
        return 1 - f_score(predicted, actual, beta=1., eps = self.eps, threshold= None, activation= self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'
    def __init__(self, eps = 1e-7, activation = 'sigmoid', lamdba_dice = 1.0, lambda_bce = 1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lamdba_dice = lamdba_dice
        self.lambda_bce = lambda_bce

    def forward(self, predicted, actual):
        dice = super().forward(predicted, actual)
        bce = self.bce(predicted, actual)
        return (self.lamdba_dice * dice) + (self.lambda_bce * bce)


