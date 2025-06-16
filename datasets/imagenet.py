import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
from collections import defaultdict
import random
from tqdm import tqdm
from .oxford_pets import OxfordPets
import pdb

@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        num_unlabeled_shots = cfg.DATASET.NUM_UNLABELED_SHOTS
        if num_shots >= 1:
            if cfg.TRAINER.STRATEGY == "supervised":
                seed = cfg.SEED
                preprocessed = os.path.join(
                    self.split_fewshot_dir, f"supervised_shot_{num_shots}-seed_{seed}.pkl"
                )

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train = data["train"]
                else:
                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    data = {"train": train}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)
                super().__init__(train_x=train, val=test, test=test)
            elif cfg.TRAINER.STRATEGY == "semi-supervised":
                seed = cfg.SEED
                ratio = 0.05  # 你的采样比例
                preprocessed = os.path.join(
                    self.split_fewshot_dir,
                    f"semi_supervised_shot_{num_shots}_unlabeled_shot_{num_unlabeled_shots}-seed_{seed}-ratio_{ratio:.2f}.pkl",
                )

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train_x, train_u = (
                            data["train_x"],
                            data["train_u"],
                        )
                        # 直接加载，无需再采样
                        num_new_train_u = len(train_u)
                        print(f"当前根据{ratio:.2f}比例采样后的train_u数量: {num_new_train_u}")
                else:
                    train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    train_u = self.generate_fewshot_dataset(
                        train, num_shots=num_unlabeled_shots
                    )                    
                    # 按照标签分组，然后每个类别取 ratio * 100% 的数据
                    print(f"train_u 中每个类别取 {ratio * 100}% 的数据")
                    num_old_train_u = len(train_u)
                    # 按标签分组
                    label_groups = defaultdict(list)
                    for item in tqdm(train_u, desc="按标签分组"):
                        label_groups[item.label].append(item)
                    new_train_u = []
                    train_x_impath = [item.impath for item in train_x]
                    for _, items in tqdm(label_groups.items(), desc="处理每个类别"):
                        # 去除重复的数据
                        items = [item for item in items if item.impath not in train_x_impath]
                        n_total = len(items)
                        if n_total == 0:
                            continue
                        n_select = int(ratio * n_total)
                        if n_select == 0:
                            n_select = 1
                        n_select = min(n_select, n_total)
                        # 随机抽样
                        selected = random.sample(items, n_select)
                        new_train_u.extend(selected)
                    train_u = new_train_u
                    num_new_train_u = len(train_u)
                    print(f"num_old_train_u: {num_old_train_u}, num_new_train_u: {num_new_train_u}")

                    # 保存采样后的train_u
                    data = {"train_x": train_x, "train_u": train_u}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                subsample = cfg.DATASET.SUBSAMPLE_CLASSES
                train_x, test = OxfordPets.subsample_classes(train_x, test, subsample=subsample)
                super().__init__(
                    train_x=train_x, train_u=train_u, val=test, test=test, cfg=cfg
                )

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
