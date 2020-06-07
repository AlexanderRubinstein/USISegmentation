# The code is adapted from https://github.com/wulalago/DAF3D

from DAF3D import DAF3D
# from DataOperate import MySet, get_data_list
# from Utils import DiceLoss, dice_ratio


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import time
import os


import sys
import os
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from DAF3D import DAF3D
from torch.backends import cudnn

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

from configs.parsing import cmd_args_parsing, args_parsing
from transforms import Resize, ToTensor
from transforms import (BatchToPILImage,
                        BatchToTensor,
                        BatchHorizontalFlip,
                        BatchRandomRotation,
                        BatchRandomScale,
                        BatchBrightContrastJitter,
                        BatchEncodeSegmentaionMap)

from dataset import SegmentationDataset, ConcatDataset, SequentialSampler, BatchSampler
from models import UNetTC, UNetTC4, UNetTC3, UNetCLSTM, UNet, Unet_with_attention, UNetFourier

from metrics import DiceMetric
from losses import CrossEntropyLoss, GeneralizedDiceLoss, CombinedLoss

from visualization import process_to_plot

cudnn.benchmark = False

class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


# for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
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

def setup_experiment(title, log_dir="./tb", experiment_name=None):
    if experiment_name is None: 
        experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%Hh%Mm%Ss"))
    writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
    best_model_path = f"{title}.best.pth"
    
    return writer, experiment_name, best_model_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_transform(batch_transform=None):
    def collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        if batch_transform is not None:
            collated = batch_transform(collated)
        return collated
    return collate

def run_epoch(model, iterator, optimizer, metric, weighted_metric=None, phase='train', epoch=0, device='cpu', writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    criterion_bce = torch.nn.BCELoss()
    criterion_dice = DiceLoss()
    
    epoch_loss = 0.0
    epoch_metric = 0.0
    if weighted_metric is not None:
        epoch_weighted_metric = 0.0
    
    with torch.set_grad_enabled(is_train):
        batch_to_plot = np.random.choice(range(len(iterator)))
        for i, (images, masks) in enumerate(tqdm(iterator)):
            images, masks = images.to(device), masks.to(device)
            
            # predicted_masks = model(images)
            
            # loss = criterion(predicted_masks, masks)



            if is_train:

                outputs1, outputs2, outputs3, outputs4, outputs1_1, outputs1_2, outputs1_3, outputs1_4, output = model(images)

                predicted_masks = output

                output = F.sigmoid(output)
                outputs1 = F.sigmoid(outputs1)
                outputs2 = F.sigmoid(outputs2)
                outputs3 = F.sigmoid(outputs3)
                outputs4 = F.sigmoid(outputs4)
                outputs1_1 = F.sigmoid(outputs1_1)
                outputs1_2 = F.sigmoid(outputs1_2)
                outputs1_3 = F.sigmoid(outputs1_3)
                outputs1_4 = F.sigmoid(outputs1_4)

                label = masks.to(torch.float)

                loss0_bce = criterion_bce(output, label)
                loss1_bce = criterion_bce(outputs1, label)
                loss2_bce = criterion_bce(outputs2, label)
                loss3_bce = criterion_bce(outputs3, label)
                loss4_bce = criterion_bce(outputs4, label)
                loss5_bce = criterion_bce(outputs1_1, label)
                loss6_bce = criterion_bce(outputs1_2, label)
                loss7_bce = criterion_bce(outputs1_3, label)
                loss8_bce = criterion_bce(outputs1_4, label)

                loss0_dice = criterion_dice(output, label)
                loss1_dice = criterion_dice(outputs1, label)
                loss2_dice = criterion_dice(outputs2, label)
                loss3_dice = criterion_dice(outputs3, label)
                loss4_dice = criterion_dice(outputs4, label)
                loss5_dice = criterion_dice(outputs1_1, label)
                loss6_dice = criterion_dice(outputs1_2, label)
                loss7_dice = criterion_dice(outputs1_3, label)
                loss8_dice = criterion_dice(outputs1_4, label)

                loss = loss0_bce + 0.4 * loss1_bce + 0.5 * loss2_bce + 0.7 * loss3_bce + 0.8 * loss4_bce + \
                    0.4 * loss5_bce + 0.5 * loss6_bce + 0.7 * loss7_bce + 0.8 * loss8_bce + \
                    loss0_dice + 0.4 * loss1_dice + 0.5 * loss2_dice + 0.7 * loss3_dice + 0.8 * loss4_dice + \
                    0.4 * loss5_dice + 0.7 * loss6_dice + 0.8 * loss7_dice + 1 * loss8_dice
            
            else:
                predict = model(images)
                predicted_masks = F.sigmoid(predict).cpu().numpy()

                predicted_masks_0 = predicted_masks <= 0.5
                predicted_masks_1 = predicted_masks > 0.5
                predicted_masks = np.concatenate([predicted_masks_0, predicted_masks_1], axis=1).astype(np.int)

                criterion = CombinedLoss([CrossEntropyLoss(), GeneralizedDiceLoss(weighted=True)], [0.4, 0.6])
                # print(predicted_masks.shape)
                # print(masks.shape)
                predicted_masks = torch.tensor(predicted_masks).to(device)
                loss = criterion(predicted_masks.to(torch.float), masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            epoch_metric += metric(torch.argmax(predicted_masks, dim=1), masks)
            if weighted_metric is not None:
                epoch_weighted_metric += weighted_metric(torch.argmax(predicted_masks, dim=1), masks)
            
            if i == batch_to_plot:
                images_to_plot, masks_to_plot, predicted_masks_to_plot = process_to_plot(images, masks, predicted_masks)

        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)
            writer.add_scalar(f"metric_epoch/{phase}", epoch_metric / len(iterator), epoch)
            if weighted_metric is not None:
                writer.add_scalar(f"weighted_metric_epoch/{phase}", epoch_weighted_metric / len(iterator), epoch)
            
            # show images from last batch

            # send to tensorboard them to tensorboard
            writer.add_images(tag='images', img_tensor=images_to_plot, global_step=epoch+1)
            writer.add_images(tag='true masks', img_tensor=masks_to_plot, global_step=epoch+1)
            writer.add_images(tag='predicted masks', img_tensor=predicted_masks_to_plot, global_step=epoch+1)

        if weighted_metric is not None:
            return epoch_loss / len(iterator), epoch_metric / len(iterator), epoch_weighted_metric / len(iterator)
        return epoch_loss / len(iterator), epoch_metric / len(iterator), None

def train(model,
          train_dataloader, val_dataloader,
          optimizer, scheduler,
          metric, weighted_metric,
          n_epochs,
          device,
          writer,
          best_model_path,
          best_checkpoint_path,
          checkpoint=None,
          new_checkpoint_path=None):

    best_val_loss = float('+inf')
    start_epoch = -1
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # torch.save(model.state_dict(), best_model_path)
    # #debug
    for epoch in range(start_epoch+1, start_epoch+n_epochs):
        train_loss, train_metric, train_weighted_metric = run_epoch(model, train_dataloader,
                                                                    optimizer,
                                                                    metric, weighted_metric,
                                                                    phase='train', epoch=epoch,
                                                                    device=device, writer=writer)
        val_loss, val_metric, val_weighted_metric = run_epoch(model, val_dataloader,
                                                              None,
                                                              metric, weighted_metric,
                                                              phase='val', epoch=epoch,
                                                              device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
            }, best_checkpoint_path)

        print(f"Best val loss so far: {best_val_loss:.3f}")

        if new_checkpoint_path is not None:
          torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'best_val_loss': best_val_loss
          }, new_checkpoint_path)

        print(f'Epoch: {epoch+1:02}')
        if weighted_metric is not None:
            print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_metric:.3f} | Train WeightedMetric: {train_weighted_metric:.3f}')
            print(f'\t  Val Loss: {val_loss:.3f} |   Val Metric: {val_metric:.3f} |   Val WeightedMetric: {val_weighted_metric:.3f}')
        else:
            print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_metric:.3f}')
            print(f'\t  Val Loss: {val_loss:.3f} |   Val Metric: {val_metric:.3f}')

def main(argv):
    params = args_parsing(cmd_args_parsing(argv))
    root, experiment_name, image_size, batch_size, lr, n_epochs, log_dir, checkpoint_path = (                                                  
        params['root'],
        params['experiment_name'],
        params['image_size'],
        params['batch_size'],
        params['lr'],
        params['n_epochs'],
        params['log_dir'],
        params['checkpoint_path']
    )

    train_val_split(os.path.join(root, DATASET_TABLE_PATH))
    dataset = pd.read_csv(os.path.join(root, DATASET_TABLE_PATH))

    pre_transforms = torchvision.transforms.Compose([Resize(size=image_size), ToTensor()])
    batch_transforms = torchvision.transforms.Compose([BatchEncodeSegmentaionMap()])
    augmentation_batch_transforms = torchvision.transforms.Compose([BatchToPILImage(),
                                                                    BatchHorizontalFlip(p=0.5),
                                                                    BatchRandomRotation(degrees=10),
                                                                    BatchRandomScale(scale=(1.0, 2.0)),
                                                                    BatchBrightContrastJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0)),
                                                                    BatchToTensor(),
                                                                    BatchEncodeSegmentaionMap()])
    
    train_dataset = SegmentationDataset(dataset=dataset[dataset['phase'] == 'train'],
                                        transform=pre_transforms)
    
    train_sampler = SequentialSampler(train_dataset)
    train_batch_sampler = BatchSampler(train_sampler, batch_size)
    train_collate = collate_transform(augmentation_batch_transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   collate_fn=train_collate)

    val_dataset = SegmentationDataset(dataset=dataset[dataset['phase'] == 'val'],
                                      transform=pre_transforms)

    val_sampler = SequentialSampler(val_dataset)
    val_batch_sampler = BatchSampler(val_sampler, batch_size)
    val_collate = collate_transform(batch_transforms)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_sampler=val_batch_sampler,
                                                 collate_fn=val_collate)
    
    model = DAF3D().to(device)

    writer, experiment_name, best_model_path = setup_experiment(model.__class__.__name__, log_dir, experiment_name)
    
    new_checkpoint_path = os.path.join(root, 'checkpoints', experiment_name  + '_latest.pth')
    best_checkpoint_path = os.path.join(root, 'checkpoints', experiment_name  + '_best.pth')
    os.makedirs(os.path.dirname(new_checkpoint_path), exist_ok=True)
    
    if checkpoint_path is not None:
        checkpoint_path = os.path.join(root, 'checkpoints', checkpoint_path)
        print(f"\nLoading checkpoint from {checkpoint_path}.\n")
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = None
    best_model_path = os.path.join(root, best_model_path)
    print(f"Experiment name: {experiment_name}")
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print()

    criterion = CombinedLoss([CrossEntropyLoss(), GeneralizedDiceLoss(weighted=True)], [0.4, 0.6])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = DiceMetric()
    weighted_metric = DiceMetric(weighted=True)
    
    print("To see the learning process, use command in the new terminal:\ntensorboard --logdir <path to log directory>")
    print()
    train(model,
          train_dataloader, val_dataloader,
          optimizer, scheduler,
          metric, weighted_metric,
          n_epochs,
          device,
          writer,
          best_model_path,
          best_checkpoint_path,
          checkpoint,
          new_checkpoint_path)

if __name__ == "__main__":
    main(sys.argv[1:])