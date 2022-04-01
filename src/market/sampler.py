from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = [
    'RandomIdentitySampler', 'SequentialSampler', 'RandomSampler',
    'RandomDomainSampler', 'RandomDatasetSampler'
]


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length



def build_train_sampler(
    data_source,
    train_sampler,
    batch_size=32,
    num_instances=4,
    num_cams=1,
    num_datasets=1,
    **kwargs
):
    """Builds a training sampler.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    elif train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)

    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)

    return sampler