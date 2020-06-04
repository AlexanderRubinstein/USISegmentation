import random
import torch
import torchvision
import torchvision.transforms.functional as TF


class Resize(object):
    def __init__(self, size):
        self.resize = torchvision.transforms.Resize(size, interpolation=2)
        
    def __call__(self, sample):
        image, mask = sample
        
        image = self.resize(image)
        mask = self.resize(mask)
        
        return (image, mask)

class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        image, mask = sample
        
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        return (image, mask)

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees)
        
    def __call__(self, sample):
        image, mask = sample
        angle = torchvision.transforms.RandomRotation.get_params(self.degrees)
        
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        
        return (image, mask)

class RandomScale(object):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, sample):
        image, mask = sample
        ret = torchvision.transforms.RandomAffine.get_params((0, 0), None, self.scale, None, image.size)
        
        image = TF.affine(image, *ret, resample=False, fillcolor=0)
        mask = TF.affine(mask, *ret, resample=False, fillcolor=0)
        
        return (image, mask)

class BrightContrastJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, sample):
        image, mask = sample
        transform = torchvision.transforms.ColorJitter(self.brightness, self.contrast, 0, 0)
        
        image = transform(image)
        
        return (image, mask)

class ToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()
    
    def __call__(self, sample):
        image, mask = sample
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return (image, mask)

class BatchToPILImage(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToPILImage()
    
    def __call__(self, batch):
        images, masks = batch
        
        batch_size = images.shape[0]
        transformed_images = []
        transformed_masks = []
        for i in range(batch_size):
            transformed_images.append(self.transform(images[i]))
            transformed_masks.append(self.transform(masks[i]))
        
        return (transformed_images, transformed_masks)

class BatchToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()
    
    def __call__(self, batch):
        images, masks = batch
        
        transformed_images = []
        transformed_masks = []
        for image, mask in zip(images, masks):
            transformed_images.append(self.transform(image))
            transformed_masks.append(self.transform(mask))
        
        return (torch.stack(transformed_images, dim=0), torch.stack(transformed_masks, dim=0))

class BatchHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, batch):
        images, masks = batch
        
        if random.random() < self.p:
            transformed_images = []
            transformed_masks = []
            for image, mask in zip(images, masks):
                transformed_images.append(TF.hflip(image))
                transformed_masks.append(TF.hflip(mask))
            return (transformed_images, transformed_masks)
        return (images, masks)

class BatchRandomRotation(object):
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees)
        
    def __call__(self, batch):
        images, masks = batch
        
        angle = torchvision.transforms.RandomRotation.get_params(self.degrees)
        transformed_images = []
        transformed_masks = []
        for image, mask in zip(images, masks):
            transformed_images.append(TF.rotate(image, angle))
            transformed_masks.append(TF.rotate(mask, angle))
        return (transformed_images, transformed_masks)

class BatchRandomScale(object):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, batch):
        images, masks = batch
        
        ret = torchvision.transforms.RandomAffine.get_params((0, 0), None, self.scale, None, images[0].size)
        transformed_images = []
        transformed_masks = []
        for image, mask in zip(images, masks):
            transformed_images.append(TF.affine(image, *ret, resample=False, fillcolor=0))
            transformed_masks.append(TF.affine(mask, *ret, resample=False, fillcolor=0))
        return (transformed_images, transformed_masks)

class BatchBrightContrastJitter(object):
    def __init__(self, brightness=None, contrast=None):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, batch):
        images, masks = batch
        
        transform = torchvision.transforms.ColorJitter().get_params(self.brightness, self.contrast, None, None)
        transformed_images = []
        for image in images:
            transformed_images.append(transform(image))
        return (transformed_images, masks)

class BatchEncodeSegmentaionMap(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):
        images, masks = batch
        
        batch_size = images.shape[0]
        transformed_masks = []
        for i in range(batch_size):
            transformed_masks.append(self.encode_segmentation_map(masks[i]))
        return (images, torch.stack(transformed_masks, dim=0))
    
    def encode_segmentation_map(self, mask): 
        labels_map = torch.zeros(mask.shape[1:])
        labels_map[mask[0, :, :] > 0] = 1

        return labels_map.to(dtype=torch.int64)
