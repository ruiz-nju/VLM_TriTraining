import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    dataset_dir = "cifar10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_cifar10.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        mkdir_if_missing(self.split_fewshot_dir)

        # Check if dataset is already downloaded
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            train, val, test = DTD.read_data(self.dataset_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        num_unlabled_shots = cfg.DATASET.NUM_UNLABELED_SHOTS
        if num_shots >= 1:
            if cfg.TRAINER.STRATEGY == "supervised":
                seed = cfg.SEED
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
                )

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train, val = data["train"], data["val"]
                else:
                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                    data = {"train": train, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train, val, test = OxfordPets.subsample_classes(
                    train, val, test, subsample=subsample
                )

                super().__init__(train_x=train, val=val, test=test)
            elif cfg.TRAINER.STRATEGY == "semi-supervised":
                seed = cfg.SEED
                preprocessed = os.path.join(
                    self.split_fewshot_dir,
                    f"semi_supervised_shot_{num_shots}_unlabeled_shot_{num_unlabled_shots}-seed_{seed}.pkl",
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
                        train, num_shots=num_unlabled_shots
                    )
                    val = self.generate_fewshot_dataset(
                        val, num_shots=min(num_shots, 4)
                    )
                    data = {"train_x": train_x, "train_u": train_u, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train_x, val, test = OxfordPets.subsample_classes(
                    train_x, val, test, subsample=subsample
                )
                # 去除重复的数据
                train_x_impath = [item.impath for item in train_x]
                train_u = [item for item in train_u if item.impath not in train_x_impath]
                super().__init__(
                    train_x=train_x, train_u=train_u, val=val, test=test, cfg=cfg
                )
