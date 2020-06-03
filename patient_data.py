import os
import shutil

import numpy as np
from PIL import Image
import cv2

from tqdm import tqdm


class Patient(object):
    """Class for preprocessing patient data."""
    def __init__(self, patient_data_path, inference=False):
        """Constructor."""
        self.patient_data_path = patient_data_path
        self.images_path = os.path.join(patient_data_path, 'Images')
        self.masks_path = None
        if not inference:
            self.masks_path = os.path.join(patient_data_path, 'Masks')
        
    def get_patient_name(self):
        return os.path.basename(self.patient_data_path)
    
    def get_images_paths(self):
        return [os.path.join(self.images_path, image_name) for image_name in sorted(os.listdir(self.images_path))]
    
    def get_masks_paths(self):
        if self.masks_path is None:
            return None
        return [os.path.join(self.masks_path, mask_name) for mask_name in sorted(os.listdir(self.masks_path))]
    
    def read_tiff_file(self, path):
        image = Image.open(path)
        images = []
        
        for frame in range(image.n_frames):
            image.seek(frame)
            images.append(np.array(image.convert("RGB")))
        
        return np.array(images)
    
    def get_patient_data(self):
        patient_data = {}
        if self.get_masks_paths() is None:
            for image_path in tqdm(self.get_images_paths()):
                name = os.path.splitext(os.path.basename(image_path))[0]
                patient_data[name] = {'image': self.read_tiff_file(image_path)}
        else:
            for image_path, mask_path in zip(tqdm(self.get_images_paths()), self.get_masks_paths()):
                name = os.path.splitext(os.path.basename(image_path))[0]
                patient_data[name] = {'image': self.read_tiff_file(image_path),
                                      'mask': self.read_tiff_file(mask_path)}
        
        return patient_data
    
    def preprocess_image(self, image):
        _, processed = cv2.threshold(image.copy(), 225, 255, cv2.THRESH_TOZERO_INV)

        return processed

    def find_largest_contour(self, image):
        processed = self.preprocess_image(image)
        contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
    
        return contour

    def show_contour(self, image, contour, colour=(0, 255, 0)):
        img = image.copy()
        cv2.drawContours(img, [contour], 0, colour, 3)
        plt.imshow(img)
    
    def get_crop_mask(self, image):
        contour = self.find_largest_contour(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], (255, 255, 255))
    
        return mask

    def clean_image(self, image, mask):
        return cv2.bitwise_and(image, mask)
    
    def data_preprocessing(self, patient_data):
        for image_id in tqdm(patient_data.keys()):
            mask = self.get_crop_mask(patient_data[image_id]['image'][0])
            for frame_number in range(patient_data[image_id]['image'].shape[0]):
                patient_data[image_id]['image'][frame_number] = self.clean_image(patient_data[image_id]['image'][frame_number], mask)
    
    def save_tif_images(self, patient_data, data_path):
        patient_name = self.get_patient_name()
        path_to_save = os.path.join(data_path, patient_name, 'Images')
        try: 
            os.makedirs(path_to_save, exist_ok=True)
        except OSError: 
            pass
        
        for image_id in tqdm(patient_data.keys()):
            images = [Image.fromarray(patient_data[image_id]['image'][i]) for i in range(patient_data[image_id]['image'].shape[0])]
            images[0].save(os.path.join(path_to_save, '{}.tif'.format(image_id)),
                           save_all=True,
                           append_images=images[1:])
            
        path_to_save = os.path.join(data_path, patient_name, 'Masks')
        if os.path.exists(path_to_save):
            shutil.rmtree(path_to_save)
        shutil.copytree(self.masks_path, path_to_save)
    
    def make_dataset_table(self, patient_data, data_path):
        patient_path = os.path.join(data_path, self.get_patient_name())
        
        data = []
        for image_id in patient_data.keys():
            for frame in range(patient_data[image_id]['image'].shape[0]):
                image_path = os.path.join(patient_path, 'Images', '{}.tif'.format(image_id))
                mask_path = os.path.join(patient_path, 'Masks', '{}.labels.tif'.format(image_id))
                
                data.append(np.array([image_path, mask_path, frame]))
            
        return np.vstack(data)
