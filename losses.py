import torch
from metrics import DiceCoefficient


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        
        self.loss = torch.nn.NLLLoss(reduction='mean')
        
    def forward(self, predicted_masks, masks):
        return self.loss(torch.log(predicted_masks), masks)

class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
        
    def forward(self, predicted_masks, masks):
        return 1 - DiceCoefficient(predicted_masks, masks)

class CombinedLoss(torch.nn.Module):
    def __init__(self, losses, coefficients=[0.4, 0.6]):
        super(CombinedLoss, self).__init__()
        
        self.losses = losses
        self.coefficients = coefficients
    
    def forward(self, predicted_masks, masks):
        loss = 0.0
        for loss_function, coefficient in zip(self.losses, self.coefficients):
            loss += coefficient * loss_function(predicted_masks, masks)
        
        return loss
