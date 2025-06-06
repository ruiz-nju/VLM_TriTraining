import numpy as np
import os.path as osp
import math
import random

from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class STL10(DatasetBase):
    """STL-10 dataset.

    Description:
    - 10 classes: airplane, bird, car, cat, deer, dog, horse,
    monkey, ship, truck.
    - Images are 96x96 pixels, color.
    - 500 training images per class, 800 test images per class.
    - 100,000 unlabeled images for unsupervised learning.

    Reference:
        - Coates et al. An Analysis of Single Layer Networks in
        Unsupervised Feature Learning. AISTATS 2011.
    """

    dataset_dir = "stl10"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train")
        test_dir = osp.join(self.dataset_dir, "test")
        unlabeled_dir = osp.join(self.dataset_dir, "unlabeled")
        fold_file = osp.join(self.dataset_dir, "stl10_binary", "fold_indices.txt")

        # Only use the first five splits
        assert 0 <= cfg.DATASET.STL10_FOLD <= 4

        train_x = self._read_data_train(train_dir, cfg.DATASET.NUM_SHOTS)
        train_u = self._read_data_all(unlabeled_dir)
        test = self._read_data_train(test_dir)

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data_train(self, data_dir, num_shots=None):
        imnames = listdir_nohidden(data_dir)
        imnames.sort()
        items = []

        list_idx = list(range(len(imnames)))
        # if fold >= 0:
        #     with open(fold_file, "r") as f:
        #         str_idx = f.read().splitlines()[fold]
        #         list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=" ")

        for label, class_name in enumerate(imnames):
            # print(label)
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            imnames_train = imnames

            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                # items.append(item)
                if num_shots is not None and (i + 1) > num_shots:
                    continue
                items.append(item)

        return items

    def _read_data_all(self, data_dir):
        imnames = listdir_nohidden(data_dir)
        items = []
        label = None
        for imname in imnames:
            impath = osp.join(data_dir, imname)
            # label = osp.splitext(imname)[0].split("_")[1]
            if label == None:
                label = -1
            else:
                label = int(label)
            item = Datum(impath=impath, label=label)
            items.append(item)

        return items
