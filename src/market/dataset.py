from __future__ import division, print_function, absolute_import
import copy
import os.path as osp
import glob
import re

from PIL import Image


def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.
    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 3:
            train = [(*items, 0) for items in train]
        if len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    # def __add__(self, other):
    #     """Adds two datasets together (only the train set)."""
    #     train = copy.deepcopy(self.train)

    #     for img_path, pid, camid, dsetid in other.train:
    #         pid += self.num_train_pids
    #         camid += self.num_train_cams
    #         dsetid += self.num_datasets
    #         train.append((img_path, pid, camid, dsetid))

    #     ###################################
    #     # Note that
    #     # 1. set verbose=False to avoid unnecessary print
    #     # 2. set combineall=False because combineall would have been applied
    #     #    if it was True for a specific dataset; setting it to True will
    #     #    create new IDs that should have already been included
    #     ###################################
    #     if isinstance(train[0][0], str):
    #         return ImageDataset(
    #             train,
    #             self.query,
    #             self.gallery,
    #             transform=self.transform,
    #             mode=self.mode,
    #             combineall=False,
    #             verbose=False
    #         )

    # def __radd__(self, other):
    #     """Supports sum([dataset1, dataset2, dataset3])."""
    #     if other == 0:
    #         return self
    #     else:
    #         return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.
        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.
        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.
        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        if self._train_only:
            return

        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid, dsetid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid, dsetid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')

class Market1501(ImageDataset):
    """Market1501.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'data/market'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data