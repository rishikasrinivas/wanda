import torch
import os
from torch.utils.data import DataLoader
from data.snli import SNLI, pad_collate
def create_dataloaders(max_data):
    root_dir="../CCE_NLI/models/DataLoaders"
    if not ('train_dataset.pth' in os.listdir(root_dir) and 'val_dataset.pth' in os.listdir(root_dir) and 'test_dataset.pth' in os.listdir(root_dir)):
        train = SNLI("data/snli_1.0", "train", max_data=max_data)
        train_loader = DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(train_loader.dataset, f'{root_dir}/train_dataset.pth')
        
        val = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train.stoi, train.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader.dataset, f'{root_dir}/val_dataset.pth')
        
        test = SNLI("data/snli_1.0", "test", max_data=max_data, vocab=(train.stoi, train.itos), unknowns=True)
        test_loader = DataLoader(
            test,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(test_loader.dataset, f'{root_dir}/test_dataset.pth')
    else:
        train_dataset = torch.load(f'{root_dir}/train_dataset.pth')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=100, 
            shuffle=True, 
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        val_dataset = torch.load(f'{root_dir}/val_dataset.pth')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=100, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        )
        
        test_dataset = torch.load(f'{root_dir}/test_dataset.pth')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        
    
    dataloaders = {
        'train': train_loader,
        'val':val_loader,
        'test': test_loader
    }
    return train_loader.dataset, val_loader.dataset,test_loader.dataset, dataloaders
