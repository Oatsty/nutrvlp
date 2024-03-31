import collections
from collections.abc import Mapping, Sequence
import torch
from torch.autograd import Variable


class Compose(object):
    """Composes several collate together.

    Args:
        transforms (list of ``Collate`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

class ListDictsToDictLists(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = self.ld_to_dl(batch)
        return batch

    def ld_to_dl(self, batch):
        if isinstance(batch[0], Mapping):
            return {key: self.ld_to_dl([d[key] for d in batch]) for key in batch[0]}
        else:
            return batch


class PadTensors(object):

    def __init__(self, value=0, use_keys=None, avoid_keys=None):
        self.value = value
        self.use_keys = use_keys or []
        self.avoid_keys = avoid_keys or []

    def __call__(self, batch):
        batch = self.pad_tensors(batch)
        return batch

    def pad_tensors(self, batch):
        if isinstance(batch, Mapping):
            out = {}
            for key, value in batch.items():
                if (key in self.use_keys) or \
                   (len(self.use_keys) == 0 and key not in self.avoid_keys):
                    out[key] = self.pad_tensors(value)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, Sequence) and torch.is_tensor(batch[0]):
            max_size = [max(item.size(i) for item in batch) for i in range(batch[0].dim())]
            max_size = torch.Size(max_size)
            n_batch = []
            for item in batch:
                if item.size() != max_size:
                    n_item = item.new(max_size).fill_(self.value)
                    # TODO: Improve this
                    if item.dim() == 1:
                        n_item[:item.size(0)] = item
                    elif item.dim() == 2:
                        n_item[:item.size(0), :item.size(1)] = item
                    elif item.dim() == 3:
                        n_item[:item.size(0), :item.size(1), :item.size(2)] = item
                    else:
                        raise ValueError
                    n_batch.append(n_item)
                else:
                    n_batch.append(item)
            return n_batch
        else:
            return batch


class StackTensors(object):

    def __init__(self, use_shared_memory=False, avoid_keys=None):
        self.use_shared_memory = use_shared_memory
        self.avoid_keys = avoid_keys or []

    def __call__(self, batch):
        batch = self.stack_tensors(batch)
        return batch

    # key argument is useful for debuging
    def stack_tensors(self, batch, key=None):
        if isinstance(batch, Mapping):
            out = {}
            for key, value in batch.items():
                if key not in self.avoid_keys:
                    out[key] = self.stack_tensors(value, key=key)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, Sequence) and torch.is_tensor(batch[0]):
            out = None
            if self.use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        else:
            return batch
