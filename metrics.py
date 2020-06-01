import torch


def DiceCoefficient(predicted_masks, masks, n_classes=2):
    smooth = 1e-16
    
    one_hot_masks = torch.nn.functional.one_hot(masks, num_classes=n_classes).permute(0, 3, 1, 2).to(predicted_masks.dtype)
    weights = 1.0 / (torch.pow(torch.sum(one_hot_masks, dim=(2, 3)), 2) + smooth)
    
    intersections = torch.sum(torch.sum(predicted_masks * one_hot_masks, dim=(2, 3)) * weights, dim=1)
    unions = torch.sum(torch.sum(predicted_masks + one_hot_masks, dim=(2, 3)) * weights, dim=1)

    dice_coefficients = 2 * (intersections + smooth) / (unions + smooth)
    return torch.mean(dice_coefficients)
