import random
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
