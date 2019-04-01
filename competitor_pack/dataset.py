from csv import DictReader
import json
import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class ColoradoDataset(Dataset):
    def __init__(self, csv_file='competitor_pack/data/colorado.csv', spec='competitor_pack/data/colorado-specs.json', dump='competitor_pack/dump.pt'):
        try:
            fh = open(dump)
            self.data = torch.load(dump)
        except FileNotFoundError:
            with open(spec) as f:
                spec = json.load(f)

            num_bits = {}
            for key in spec:
                maxval = spec[key]['maxval']
                num_bits[key] = 1 if maxval == 0 else math.ceil(math.log(spec[key]['maxval'] + 1, 2))

            data = []
            with open(csv_file) as f:
                reader = DictReader(f)
                for row in reader:
                    data_row = []
                    for key, value in sorted(row.items()):
                        bit_vector = list(map(int, format(int(value), '0{}b'.format(num_bits[key]))))
                        data_row.extend(bit_vector)
                    data.append(data_row)
            self.data = torch.Tensor(data).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            torch.save(self.data, dump)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print('Loading data...')
    dataset = Colorado()
    print('Done.')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, sample in enumerate(dataloader):
        print(i, sample)

