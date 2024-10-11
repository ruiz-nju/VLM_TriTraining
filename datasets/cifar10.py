import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from dassl.utils import listdir_nohidden
import math
import random
import pdb
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "CIFAR10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_train_dir = os.path.join(self.dataset_dir, "train")
        self.image_test_dir = os.path.join(self.dataset_dir, "test")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train, val, test = self.read_and_split_data()
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            # 有监督学习
            if cfg.TRAINER.STRATEGY == "supervised":
                seed = cfg.SEED
                preprocessed = os.path.join(
                    self.split_fewshot_dir,
                    f"supervised_shot_{num_shots}-seed_{seed}.pkl",
                )

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train_x, val = data["train_x"], data["val"]
                else:
                    train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(
                        val, num_shots=min(num_shots, 4)
                    )
                    data = {"train_x": train_x, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train_x, val, test = OxfordPets.subsample_classes(
                    train_x, val, test, subsample=subsample
                )
                super().__init__(train_x=train_x, val=val, test=test)
            elif cfg.TRAINER.STRATEGY == "semi-supervised":
                # 半监督学习
                seed = cfg.SEED
                preprocessed = os.path.join(
                    self.split_fewshot_dir,
                    f"semi_supervised_shot_{num_shots}-seed_{seed}.pkl",
                )

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train_x, train_u, val = (
                            data["train_x"],
                            data["train_u"],
                            data["val"],
                        )
                else:
                    train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    train_u = self.generate_fewshot_dataset(
                        train, num_shots=cfg.DATASET.NUM_SHOT_UNLABELED
                    )
                    val = self.generate_fewshot_dataset(
                        val, num_shots=min(num_shots, 4)
                    )
                    data = {"train_x": train_x, "train_u": train_u, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train_x, train_u, val, test = OxfordPets.subsample_classes(
                    train_x, train_u, val, test, subsample=subsample
                )
                super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def read_and_split_data(self, p_trn=0.5, p_val=0.2):

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for dir in [self.image_train_dir, self.image_test_dir]:
            categories = listdir_nohidden(dir)
            categories = [c for c in categories]
            # 按照默认的字典序升序标号
            categories.sort()
            # print(
            #     categories
            # )  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            # pdb.set_trace()
            for label, category in enumerate(categories):
                category_dir = os.path.join(dir, category)
                images = listdir_nohidden(category_dir)
                images = [os.path.join(category_dir, im) for im in images]
                random.shuffle(images)
                n_total = len(images)
                n_train = round(n_total * p_trn)
                n_val = round(n_total * p_val)
                n_test = n_total - n_train - n_val
                assert n_train > 0 and n_val > 0 and n_test > 0

                train.extend(_collate(images[:n_train], label, category))
                val.extend(_collate(images[n_train : n_train + n_val], label, category))
                test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
