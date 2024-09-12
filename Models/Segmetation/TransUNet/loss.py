import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, eps=1e-15):
        super(Loss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

        self.eps = eps

    def dice_loss(self, prediction, mask):
        intersection = torch.sum(prediction * mask)
        union = torch.sum(prediction * prediction) + torch.sum(mask * mask)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice

    def forward(self, prediction, mask):
        ce_loss = self.cross_entropy(prediction, mask)

        prediction = torch.softmax(prediction, dim=1)
        num_classes = prediction.shape[1]

        dice_loss = 0.0
        for i in range(num_classes):
            dice_loss += self.dice_loss(prediction[:, i], mask[:, i])

        return 0.5 * dice_loss / num_classes + 0.5 * ce_loss
