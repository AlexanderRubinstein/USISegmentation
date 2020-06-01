import sys
import os
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

from configs.parsing import cmd_args_parsing, args_parsing
from transforms import Resize, HorizontalFlip, RandomRotation, RandomScale, BrightContrastJitter, ToTensor
from dataset import SegmentationDataset
from models import UNet
from metrics import DiceCoefficient
from losses import CrossEntropyLoss, SoftDiceLoss, CombinedLoss


# for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_TABLE_PATH = './dataset.csv'

def train_val_split(csv_file_path, val_size=0.2):
    """Splitting into train and test parts."""
    dataset = pd.read_csv(csv_file_path)
    
    test_number = int(len(dataset) * val_size) + 1
    train_number = len(dataset) - test_number
    phase = ['train'] * train_number + ['val'] * test_number
    
    pd.concat([dataset[['image', 'mask', 'frame']],
               pd.DataFrame(phase, columns=['phase'])],
               axis=1).to_csv(csv_file_path, index=False)

def setup_experiment(title, log_dir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%Hh%Mm%Ss"))
    writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
    best_model_path = f"{title}.best.pth"
    
    return writer, experiment_name, best_model_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(model, iterator, criterion, optimizer, metric, phase='train', epoch=0, device='cpu', writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0.0
    epoch_metric = 0.0
    
    with torch.set_grad_enabled(is_train):
        for i, (images, masks) in enumerate(tqdm(iterator)):
            images, masks = images.to(device), masks.to(device)
            
            predicted_masks = model(images)
            
            loss = criterion(predicted_masks, masks)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            epoch_metric += metric(predicted_masks, masks)    

        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)
            writer.add_scalar(f"metric_epoch/{phase}", epoch_metric / len(iterator), epoch)

        return epoch_loss / len(iterator), epoch_metric / len(iterator)

def train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs,
          device,
          writer,
          best_model_path):

    best_val_loss = float('+inf')
    for epoch in range(n_epochs):
        train_loss, train_metric = run_epoch(model, train_dataloader,
                                             criterion, optimizer, metric,
                                             phase='train', epoch=epoch,
                                             device=device, writer=writer)
        val_loss, val_metric = run_epoch(model, val_dataloader,
                                         criterion, None, metric,
                                         phase='val', epoch=epoch,
                                         device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_metric:.3f}')
        print(f'\t  Val Loss: {val_loss:.3f} |   Val Metric: {val_metric:.3f}')

def main(argv):
    params = args_parsing(cmd_args_parsing(argv))
    root, image_size, batch_size, n_epochs, log_dir = params['root'], params['image_size'], params['batch_size'], params['n_epochs'], params['log_dir']
    
    train_val_split(os.path.join(root, DATASET_TABLE_PATH))
    dataset = pd.read_csv(os.path.join(root, DATASET_TABLE_PATH))
    
    augmentation_transforms = [HorizontalFlip(),
                               torchvision.transforms.Compose([RandomRotation(10), RandomScale(scale=(1.1, 1.5))]),
                               BrightContrastJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0))]
    
    transforms = torchvision.transforms.Compose([Resize(image_size), ToTensor()])
    train_dataset = SegmentationDataset(dataset=dataset[dataset['phase'] == 'train'],
                                        transform=transforms)
    for transform in augmentation_transforms:
        transforms = torchvision.transforms.Compose([Resize(image_size),
                                                     transform,
                                                     ToTensor()])
        augmented_dataset = SegmentationDataset(dataset=dataset[dataset['phase'] == 'train'],
                                                transform=transforms)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmented_dataset])

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    #                                                num_workers=4)

    val_dataset = SegmentationDataset(dataset=dataset[dataset['phase'] == 'val'],
                                      transform=transforms)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    #                                              num_workers=4)
    
    model = UNet(1, 2).to(device)

    writer, experiment_name, best_model_path = setup_experiment(model.__class__.__name__, log_dir)
    best_model_path = os.path.join(root, best_model_path)
    print(f"Experiment name: {experiment_name}")
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print()

    criterion = CombinedLoss([CrossEntropyLoss(), SoftDiceLoss()], [0.4, 0.6])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = DiceCoefficient
    
    print("To see the learning process, use command in the new terminal:\ntensorboard --logdir <path to log directory>")
    print()
    train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs,
          device,
          writer,
          best_model_path)

if __name__ == "__main__":
    main(sys.argv[1:])
