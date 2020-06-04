import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, predicted_masks, masks):
        return self.loss(predicted_masks, masks)

class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, n_classes=2, weighted=False):
        super(GeneralizedDiceLoss, self).__init__()
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.weighted = weighted
        
    def forward(self, predicted_masks, masks):
        predicted_masks = self.softmax(predicted_masks)
        smooth = 1e-16
    
        one_hot_masks = torch.nn.functional.one_hot(masks, num_classes=self.n_classes)
        one_hot_masks = one_hot_masks.permute(0, 3, 1, 2).to(predicted_masks.dtype)

        if self.weighted:
            weights = 1.0 / (torch.pow(torch.sum(one_hot_masks, dim=(2, 3)), 2) + smooth)
        else:
            weights = torch.ones(predicted_masks.shape[:2]).to(predicted_masks.device)
        intersections = torch.sum(torch.sum(predicted_masks * one_hot_masks, dim=(2, 3)) * weights, dim=1)
        unions = torch.sum(torch.sum(predicted_masks + one_hot_masks, dim=(2, 3)) * weights, dim=1)

        return 1 - torch.mean(2 * (intersections + smooth) / (unions + smooth))

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
