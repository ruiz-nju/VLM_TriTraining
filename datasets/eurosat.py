import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):

    dataset_dir = "eurosat"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(
                self.image_dir, new_cnames=NEW_CNAMES
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]  # type: ignore
            item_new = Datum(
                impath=item_old.impath, label=item_old.label, classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
