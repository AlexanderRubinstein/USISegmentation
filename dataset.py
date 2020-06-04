import torch
from PIL import Image
import bisect
from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data.sampler import SequentialSampler, BatchSampler


class SegmentationDataset(Dataset):
    """Class for torch Dataset forming."""
    def __init__(self, dataset, transform=None):
        """Constructor."""
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.dataset.iloc[index]['image']
        mask_path = self.dataset.iloc[index]['mask']
        frame = self.get_frame(index)
        
        image = self.read_tiff_file_frame(image_path, frame)
        mask = self.read_tiff_file_frame(mask_path, frame)

        if self.do_transform():
            image, mask = self.transform((image, mask))
        
        return (image, mask)

    def __len__(self):
        return len(self.dataset)
    
    def get_frame(self, index):
        return self.dataset.iloc[index]['frame']
    
    def read_tiff_file_frame(self, path, frame):
        image = Image.open(path)
        image.seek(frame)
        
        return image.convert("L")
    
    def do_transform(self):
        return self.transform is not None

class ConcatDataset(ConcatDataset):
    def get_frame(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index].get_frame(sample_index)


class SequentialSampler(SequentialSampler):
    def __iter__(self):
        indices = [(index, self.data_source.get_frame(index)) for index in range(len(self.data_source))]
        return iter(indices)

class BatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        prev_frame = 0
        for index, frame in self.sampler:
            if frame < prev_frame and len(batch) != 0:
                batch = [batch[-1] - len(batch) - i for i in range(self.batch_size - len(batch))][::-1] + batch
                yield batch
                batch = []
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            prev_frame = frame
        if len(batch) > 0:
            batch = [batch[-1] - len(batch) - i for i in range(self.batch_size - len(batch))] + batch
            yield batch

    def __len__(self):
        init_index = 0
        length = 0
        for index, frame in self.sampler:
            if frame == 0:
                length += (index - init_index) // self.batch_size
                if (index - init_index) % self.batch_size != 0:
                    length += 1
                init_index = index
        index += 1
        length += (index - init_index) // self.batch_size
        if (index - init_index) % self.batch_size != 0:
            length += 1
        
        return length
