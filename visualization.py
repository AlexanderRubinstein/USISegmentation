# here functions to visualize images and masks
# during train cycle are represented
import torch
import numpy as np

def process_to_plot(images, masks, predicted_masks):
    with torch.no_grad(): # just in case
        # to numpy
        images = images.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        predicted_masks = torch.argmax(predicted_masks, dim=1).detach().cpu().numpy()

        # do the reshape to make "channel" dimension
        batch_size = masks.shape[0]
        height = masks.shape[1]
        width = masks.shape[2]

        images = np.reshape(images, (batch_size, 1, height, width))
        masks = np.reshape(masks, (batch_size, 1, height, width))
        predicted_masks = np.reshape(predicted_masks, (batch_size, 1, height, width))

        return images, masks, predicted_masks


