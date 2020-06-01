import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class SegmentationDataset(Dataset):
    """Class for torch Dataset forming."""
    def __init__(self, dataset, transform=None):
        """Constructor."""
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.dataset.iloc[index]['image']
        mask_path = self.dataset.iloc[index]['mask']
        frame = self.dataset.iloc[index]['frame']
        
        image = self.read_tiff_file_frame(image_path, frame)
        mask = self.read_tiff_file_frame(mask_path, frame)

        if self.do_transform():
            image, mask = self.transform((image, mask))
        
        mask = self.encode_segmentation_map(mask)
        
        return (image, mask)

    def __len__(self):
        return len(self.dataset)
    
    def read_tiff_file_frame(self, path, frame):
        image = Image.open(path)
        image.seek(frame)
        
        return image.convert("L")
    
    def do_transform(self):
        return self.transform is not None
    
    def encode_segmentation_map(self, mask): 
        labels_map = torch.zeros(mask.shape[1:])
        labels_map[mask[0, :, :] > 0] = 1

        return labels_map.to(dtype=torch.int64)
