import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split

class AMP_emb_Dataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)
        self.len = self.data['seq_emb'].shape[0]

    def __getitem__(self, index):
        return {
            'seq_emb': self.data['seq_emb'][index],        # [51, 1024]
            'text_emb': self.data['text_emb'][index],      # [35, 1024]
            'len_emb': self.data['length'][index],         # [1, 1024]
            'mask': self.data['mask'][index],
            'ids': self.data['ids'][index]
        }

    def __len__(self):
        return self.len

def get_train_test_loader(embedding_dir, bs):

    dataset = AMP_emb_Dataset(embedding_dir)

    train_size = int(0.9 * len(dataset))  
    test_size = len(dataset) - train_size  

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, test_loader