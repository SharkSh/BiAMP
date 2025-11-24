import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split

class AMP_emb_Dataset(Dataset):
    def __init__(self, file_path):
        """
        自定义 PyTorch 数据集类，用于加载 seq_text 数据。

        :param file_path: 存放 .pt 文件的目录
        """
        self.data = torch.load(file_path)
        self.len = self.data['seq_emb'].shape[0]

    def __getitem__(self, index):
        """返回指定索引的数据"""
        return {
            'seq_emb': self.data['seq_emb'][index],        # [51, 1024]
            'text_emb': self.data['text_emb'][index],      # [35, 1024]
            'len_emb': self.data['length'][index],         # [1, 1024]
        }

    def __len__(self):
        """返回数据集大小"""
        return self.len

def get_train_test_loader(embedding_dir, bs):

    # 加载数据集
    dataset = AMP_emb_Dataset(embedding_dir)

    # 计算划分大小
    train_size = int(0.9 * len(dataset))  # 90% 训练集
    test_size = len(dataset) - train_size  # 10% 测试集

    # 按照 9:1 分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, test_loader