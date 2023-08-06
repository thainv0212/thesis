import torch


class Cutoff:
    def __call__(self, data):
        return data[:800, :]

class Cutoff2:
    def __call__(self, data):
        return data[:, :800]

class Transpose:
    def __call__(self, data):
        return data.T


class ToTensor:
    def __call__(self, data):
        return torch.Tensor(data)


class Flatten:
    def __call__(self, data):
        return torch.flatten(data)


class Mask:
    def __init__(self, selected_idx):
        print('init mask', selected_idx)
        mask = [False] * 143
        for i in selected_idx:
            mask[i] = True
        self.mask = mask

    def __call__(self, data):
        return data[self.mask]
