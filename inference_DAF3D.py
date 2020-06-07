import sys
import os
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from configs.parsing import cmd_args_parsing, args_parsing
from models import UNet
from patient_data import Patient

from torch.nn import functional as F
import numpy as np

from DAF3D import DAF3D


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Segmentation(object):
    def __init__(self, inference_model, checkpoint_path, image_size, device):
        self.device = device

        self.model = inference_model
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.image_size = image_size
    
    def predicted_classes(self, predicted_mask):
        return torch.argmax(predicted_mask, dim=1, keepdim=True).to(dtype=predicted_mask.dtype)
    
    def save_tif_file(self, frames, path_to_save):
        frames[0].save(path_to_save, save_all=True, append_images=frames[1:], dpi=(600, 600), optimize=False, duration=100, loop=0)
    
    def horizontal_concatenation(self, image1, image2):
        concat_image = Image.new('RGB', (image1.width + image2.width, image1.height))
        concat_image.paste(image1, (0, 0))
        concat_image.paste(image2, (image1.width, 0))
        
        return concat_image
    
    def save_gif_file(self, images, masks, predicted_masks, path_to_save):
        concat_images = [self.horizontal_concatenation(self.horizontal_concatenation(image, mask), predicted_mask) 
                         for image, mask, predicted_mask in zip(images, masks, predicted_masks)]
        concat_images[0].save(path_to_save, save_all=True, append_images=concat_images[1:])
    
    def segmentation(self, images):
        init_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.ToTensor()
        ])
        end_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((images[0].height, images[0].width))
        ])
        masks = []
        with torch.no_grad():
            for image in images:
                predict = self.model(init_transform(image).unsqueeze(0).to(self.device))
                predicted_mask = F.sigmoid(predict).cpu().numpy()

                predicted_mask_0 = 1 - predicted_mask
                predicted_mask_1 = predicted_mask
                predicted_mask = np.concatenate([predicted_mask_0, predicted_mask_1], axis=1)
                # predicted_mask = self.model(init_transform(image).unsqueeze(0).to(self.device))
                predicted_mask = torch.tensor(predicted_mask).to(device)
                mask = self.predicted_classes(predicted_mask)
                masks.append(mask.squeeze(0).cpu())
        
        return [end_transform(mask) for mask in masks]

def main(argv):
    params = args_parsing(cmd_args_parsing(argv))
    model_path, image_size, test_data_path = params['model_path'], params['image_size'], params['test_data_path']
    
    # segmentator = Segmentation(UNet(1, 2), model_path, image_size, device)
    segmentator = Segmentation(DAF3D(), model_path, image_size, device)
    
    patients_paths = [os.path.join(test_data_path, patient_name) for patient_name in os.listdir(test_data_path)]

    print('segmentation masks prediction for {} patients'.format(len(patients_paths)))
    print()

    dataset = []
    for patient_path in patients_paths:
        patient = Patient(patient_path, inference=False)
        patient_name = patient.get_patient_name()

        print('{} data reading ...'.format(patient_name)) 
        patient_data = patient.get_patient_data()

        print('{} data preprocessing ...'.format(patient_name))
        patient.data_preprocessing(patient_data)

        print('{} data segmentation ...'.format(patient_name))
        for image_id in tqdm(patient_data.keys()):
            images = [Image.fromarray(patient_data[image_id]['image'][i]).convert("L")
                      for i in range(patient_data[image_id]['image'].shape[0])]
            masks = [Image.fromarray(patient_data[image_id]['mask'][i]).convert("RGB")
                      for i in range(patient_data[image_id]['mask'].shape[0])]
            predicted_masks = segmentator.segmentation(images)

            path_to_save = os.path.join(test_data_path, patient_name, 'Predicted_Masks')
            try:
                os.makedirs(path_to_save, exist_ok=True)
            except OSError: 
                pass
            segmentator.save_tif_file(predicted_masks, os.path.join(path_to_save, '{}.tif'.format(image_id)))

            path_to_save = os.path.join(test_data_path, patient_name, 'Animations')
            try:
                os.makedirs(path_to_save, exist_ok=True)
            except OSError: 
                pass
            segmentator.save_gif_file(images, masks, predicted_masks, os.path.join(path_to_save, '{}.gif'.format(image_id)))
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
