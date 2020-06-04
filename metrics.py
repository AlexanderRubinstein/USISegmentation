import torch


class DiceMetric(object):
    def __init__(self, n_classes=2, weighted=False):
        self.n_classes = n_classes
        self.weighted = weighted
        
    def __call__(self, predicted_masks, masks):
        smooth = 1e-16
        
        one_hot_predicted_masks = torch.nn.functional.one_hot(predicted_masks, num_classes=self.n_classes)
        one_hot_predicted_masks = one_hot_predicted_masks.permute(0, 3, 1, 2).to(predicted_masks.dtype)
        one_hot_masks = torch.nn.functional.one_hot(masks, num_classes=self.n_classes)
        one_hot_masks = one_hot_masks.permute(0, 3, 1, 2).to(predicted_masks.dtype)

        if self.weighted:
            weights = 1.0 / (torch.pow(torch.sum(one_hot_masks, dim=(2, 3)), 2) + smooth)
        else:
            weights = torch.ones(one_hot_masks.shape[:2]).to(predicted_masks.device)

        intersections = torch.sum(torch.sum(one_hot_predicted_masks * one_hot_masks, dim=(2, 3))*weights, dim=1)
        unions = torch.sum(torch.sum(one_hot_predicted_masks + one_hot_masks, dim=(2, 3))*weights, dim=1)

        dice_coefficients = 2 * (intersections + smooth) / (unions + smooth)
        return torch.mean(dice_coefficients)