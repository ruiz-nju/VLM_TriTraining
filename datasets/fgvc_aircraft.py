import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        num_shots = cfg.DATASET.NUM_SHOTS
        num_unlabled_shots = cfg.DATASET.NUM_UNLABELED_SHOTS
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
                super().__init__(
                    train_x=train_x, train_u=train_u, val=val, test=test, cfg=cfg
                )

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
