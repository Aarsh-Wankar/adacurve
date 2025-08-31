import argparse
import torch
import random
import numpy as np
import csv
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from adahessian import Adahessian
from adacurve import AdaCurve
from torchvision.datasets import ImageFolder
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split, Subset
import os

import torch.nn as nn
import torch.optim as optim

# Optional: Import custom optimizers if available
# try:
#     pass
# except ImportError:
#     Adahessian = None
# try:
#     pass
# except ImportError:
#     AdaCurve = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# MODIFIED get_dataset function
def get_dataset(name, data_dir='./data', augmentation="none"):
    if name == "cifar10":
        normalize_factor = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif name == "cifar100":
        normalize_factor = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif name == "svhn":
        normalize_factor = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    elif name == "imagenette":
        normalize_factor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        normalize_factor = transforms.Normalize((0.5,), (0.5,))
    if augmentation in ["basic", "advanced"]:
        transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
        if augmentation  == "advanced":
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        transform_train = transforms.Compose(transform_list + [transforms.ToTensor(), normalize_factor])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), normalize_factor])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_factor
    ])

    # Load the full training data first
    if name == 'cifar10':
        train_data_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        val_data_no_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_val_test)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_val_test)
        num_classes = 10
    elif name == 'cifar100':
        train_data_aug = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        val_data_no_aug = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_val_test)
        test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_val_test)
        num_classes = 100
    elif name == 'imagenette':
        train_data_aug = ImageFolder(f'{data_dir}/imagenette2/train', transform=transform_train)
        val_data_no_aug = ImageFolder(f'{data_dir}/imagenette2/val', transform=transform_val_test)
        test_set = ImageFolder(f'{data_dir}/imagenette2/val', transform=transform_val_test)
        num_classes = 10
    elif name == 'svhn':
        train_data_aug = datasets.SVHN(data_dir, split='train', download=True, transform=transform_train)
        val_data_no_aug = datasets.SVHN(data_dir, split='train', download=True, transform=transform_val_test)
        test_set = datasets.SVHN(data_dir, split='test', download=True, transform=transform_val_test)
        num_classes = 10
    else:
        raise ValueError('Unknown dataset')

    # --- SPLIT THE TRAINING DATA --- #
    train_size = int(0.9 * len(train_data_aug))
    val_size = len(train_data_aug) - train_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_data_aug), generator=generator).tolist()
    train_set = Subset(train_data_aug, indices[:train_size])
    val_set = Subset(train_data_aug, indices[train_size:])

    # train_set, val_set = random_split(train_data_aug, [train_size, val_size])
    # ----------------------------- #

    return train_set, val_set, test_set, num_classes # <-- MODIFIED: return val_set too

def get_model(name, num_classes):
    if name == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
    elif name == 'resnet20':
        raise NotImplementedError('ResNet20 is not implemented. Please provide your own implementation.')
    else:
        raise ValueError('Unknown model')
    return model

def get_optimizer(name, params, lr):
    if name == 'adam':
        return optim.Adam(params, lr=lr)
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr)
    elif name == 'adahessian':
        if Adahessian is None:
            raise ImportError('Adahessian optimizer not installed')
        return Adahessian(params, lr=lr)
    elif name == 'adacurve':
        if AdaCurve is None:
            raise ImportError('AdaCurve optimizer not installed')
        return AdaCurve(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer')
    

def get_scheduler(name, optimizer, num_epochs):
    if name == 'none':
        return None
    elif name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif name == 'step':
        # Decays the LR by a factor of 0.1 every 10 epochs
        return lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError('Unknown scheduler')

# MODIFIED train function
def train(model, device, train_loader, optimizer, criterion, epoch, step_writer): # <-- writer is now step_writer
    model.train()
    running_loss = 0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if isinstance(optimizer, (Adahessian, AdaCurve)):
            loss.backward(create_graph=True)
        else:
            loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # --- Accuracy Calculation ---
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        # Log train loss PER STEP to the dedicated step file
        step_writer.writerow({
            'epoch': epoch,
            'step': batch_idx,
            'train_loss_step': loss.item(),
        })

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / processed
    return avg_loss, train_accuracy

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    return avg_val_loss, val_accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0 # <-- ADDED: counter for correct predictions
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            # --- Accuracy Calculation --- # <-- ADDED
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # --------------------------- #

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset) # <-- ADDED
    return avg_test_loss, test_accuracy # <-- MODIFIED: return accuracy as well

def main(args=None):
    parser = argparse.ArgumentParser(description='Standard Training Pipeline')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'imagenette', 'svhn'])
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet20'])
    parser.add_argument('--optimizer', type=str, required=True, choices=['adam', 'adamw', 'adahessian', 'adacurve'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-dir', type=str, default='./experiment_results')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'])
    parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'basic', 'advanced'])

    if args is None:
        args = parser.parse_args()   # CLI call
    else:
        args = parser.parse_args(args)  # list of strings for programmatic call

    set_seed(args.seed)

    device = torch.device(args.device)
    train_set, val_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir, augmentation=args.augmentation)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    model = get_model(args.model, num_classes).to(device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)
    
    criterion = nn.CrossEntropyLoss()

    # Prepare CSV file
    base_filename = f'{args.dataset}_{args.model}_{args.optimizer}_epochs_{args.epochs}_lr_{args.lr}_augment_{args.augmentation}_seed{args.seed}'
    step_csv_filename = f'{args.log_dir}/results_step_{base_filename}.csv'
    epoch_csv_filename = f'{args.log_dir}/results_epoch_{base_filename}.csv'
    os.makedirs(args.log_dir, exist_ok=True)  # Ensure log directory exists
    with open(step_csv_filename, mode='w', newline='') as step_file, \
         open(epoch_csv_filename, mode='w', newline='') as epoch_file:

        # Setup for the STEP file
        step_fieldnames = ['epoch', 'step', 'train_loss_step']
        step_writer = csv.DictWriter(step_file, fieldnames=step_fieldnames)
        step_writer.writeheader()

        # Setup for the EPOCH file
        epoch_fieldnames = ['epoch', 'train_loss_epoch', 'val_loss_epoch', 'test_loss_epoch', 'train_accuracy_epoch', 'val_accuracy_epoch', 'test_accuracy_epoch']
        epoch_writer = csv.DictWriter(epoch_file, fieldnames=epoch_fieldnames)
        epoch_writer.writeheader()
        max_val_acc = 0.0
        best_epoch = 0
        test_acc_at_maxval = 0
        for epoch in range(1, args.epochs + 1):
            # Pass the step_writer to the train function
            avg_train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, step_writer)
            avg_val_loss, val_acc = validate(model, device, val_loader, criterion)
            avg_test_loss, test_acc = test(model, device, test_loader, criterion)

            if(val_acc > max_val_acc):
                max_val_acc = max(max_val_acc, val_acc)
                best_epoch = epoch
                test_acc_at_maxval = test_acc

            if scheduler is not None:
                scheduler.step()
            # Log summary metrics PER EPOCH to the dedicated epoch file
            epoch_writer.writerow({
                'epoch': epoch,
                'train_loss_epoch': avg_train_loss,
                'val_loss_epoch': avg_val_loss, 
                'test_loss_epoch': avg_test_loss,
                'train_accuracy_epoch': train_acc,
                'val_accuracy_epoch': val_acc,
                'test_accuracy_epoch': test_acc
            })
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.6f}, Val Acc={val_acc:.2f}%, Test Loss={avg_test_loss:.6f}, Test Acc={test_acc:.2f}%, LR={current_lr:.6f}')
        return max_val_acc, test_acc_at_maxval
if __name__ == '__main__':
    main()