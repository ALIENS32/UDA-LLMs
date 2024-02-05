from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json


# 读取json文件
def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


class Datasets(Dataset):
    def __init__(self, fp):
        # 返回值是一个列表，每一个元素是一个字典，形如：{'src': '本院认证如下：...', 'label': 1}
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['src'], self.data[index]['label']


def get_loader_and_length(fp, batch_size=2, shuffle=True, num_workers=1):
    dataset = Datasets(fp)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(dataset)
