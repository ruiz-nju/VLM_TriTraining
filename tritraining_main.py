import argparse
import sklearn.utils
import torch
import random
from torchvision.transforms import InterpolationMode
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import copy
import numpy as np
import sklearn
from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from dassl.utils import read_image
from dassl.data.transforms.transforms import build_transform
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.ToImage import ToImage
from sklearn.pipeline import Pipeline
import torchvision.transforms as transforms
from PIL import Image
from dassl.data.transforms.transforms import Cutout
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.ToImage import ToImage
from tqdm import tqdm

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.cifar10
from LAMDA_SSL.Split.DataSplit import DataSplit
import trainers.coop
import trainers.cocoop
import trainers.tri_training
import trainers.zsclip
import trainers.maple
import trainers.vpt
from trainers.tri_training import Tri_Training
from LAMDA_SSL.Transform.Vision.Resize import Resize
from torch.utils.data import DataLoader, TensorDataset

import pdb


def setup_cfg(args):
    print("----------Build up cfg----------")
    cfg = {
        "CoOp": get_cfg_default(),
        "VPT": get_cfg_default(),
        "MaPLe": get_cfg_default(),
    }
    for key in cfg.keys():
        extend_cfg(cfg[key])
        cfg[key].merge_from_file(args.dataset_config_file)
        cfg[key].merge_from_file(
            f"/mnt/hdd/zhurui/code/TriTraining/configs/trainers/TriTraining/{key}.yaml"
        )
        if args.root:
            cfg[key].DATASET.ROOT = args.root
        if args.output_dir:
            cfg[key].OUTPUT_DIR = args.output_dir
        if args.resume:
            cfg[key].RESUME = args.resume
        if args.seed:
            cfg[key].SEED = args.seed
        if args.source_domains:
            cfg[key].DATASET.SOURCE_DOMAINS = args.source_domains
        if args.target_domains:
            cfg[key].DATASET.TARGET_DOMAINS = args.target_domains
        if args.transforms:
            cfg[key].INPUT.TRANSFORMS = args.transforms
        if args.backbone:
            cfg[key].MODEL.BACKBONE.NAME = args.backbone
        if args.head:
            cfg[key].MODEL.HEAD.NAME = args.head
        cfg[key].merge_from_list(args.opts)
        cfg[key].freeze()
    return cfg


def extend_cfg(cfg):
    cfg.TRAINER.MODAL = "classification"
    from yacs.config import CfgNode as CN

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for CoCoOp
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = (
        9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    )

    # Config for VPT
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = (
        1  # if set to 1, will represent shallow vision prompting only
    )

    # 默认的采样设置为 all，可以根据需要进行调整
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def get_dataset(model):
    train_x_list = []
    train_y_list = []

    print("Get train dataset")
    for batch in tqdm(model.train_loader_x):
        # Parse batch
        X, y = model.parse_batch_train(batch)

        # Move X and y to CPU
        X_cpu = X.cpu()
        y_cpu = y.cpu()

        # Append to list
        train_x_list.append(X_cpu)
        train_y_list.append(y_cpu)

    # Concatenate tensors
    train_x_tensor = torch.cat(train_x_list, dim=0)
    train_y_tensor = torch.cat(train_y_list, dim=0)

    # Convert to NumPy arrays
    train_x = train_x_tensor.numpy()
    train_y = train_y_tensor.numpy()

    labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(
        X=train_x,
        y=train_y,
        size_split=0.1,
        random_state=1,
    )
    # 获取 test
    test_X_list = []
    test_y_list = []
    print("Get test dataset")
    for batch in tqdm(model.test_loader):
        # 解析 batch
        X, y = model.parse_batch_train(batch)

        # 将 X 和 y 从 CUDA 移动到 CPU 并转换为 NumPy 数组
        X_np = X.cpu()
        y_np = y.cpu()

        # 将 NumPy 数组添加到列表中
        test_X_list.append(X_np)
        test_y_list.append(y_np)
    test_X_tensor = torch.cat(test_X_list, dim=0)
    test_y_tensor = torch.cat(test_y_list, dim=0)
    test_X = test_X_tensor.numpy()
    test_y = test_y_tensor.numpy()
    print(f"labeled_X: {labeled_X.shape}")
    print(f"labeled_y: {labeled_y.shape}")
    print(f"unlabeled_X: {unlabeled_X.shape}")
    print(f"test_X: {test_X.shape}")
    print(f"test_y: {test_y.shape}")
    return labeled_X, labeled_y, unlabeled_X, test_X, test_y


def main(args):
    cfg = setup_cfg(args)
    base_cfg = cfg["CoOp"]

    if base_cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(base_cfg.SEED))
        set_random_seed(base_cfg.SEED)
    setup_logger(base_cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and base_cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("----------Build up CoOp----------")
    print(f"Config of CoOp:\n{cfg['CoOp']}")
    coop = build_trainer(cfg["CoOp"])

    print("----------Build up VPT----------")
    print(f"Config of VPT:\n{cfg['VPT']}")
    vpt = build_trainer(cfg["VPT"])

    print("----------Build up MaPLe----------")
    print(f"Config of MaPLe:\n{cfg['MaPLe']}")
    maple = build_trainer(cfg["MaPLe"])

    labeled_X, labeled_y, unlabeled_X, test_X, test_y = get_dataset(coop)

    # # 实例化 Tri_Training 并进行训练和测试
    tri_trainer = Tri_Training(coop, vpt, maple)

    tri_trainer.fit(labeled_X, labeled_y, unlabeled_X)

    y_pred = tri_trainer.predict(test_X)

    # 计算准确率
    acc = Accuracy()
    print("Accuracy: ", acc.score(y_pred, test_y))


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )  # 创建一个 ArgumentParser 对象，用来处理命令行输入
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )  # 1
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )  # configs/trainers/CoOp/vit_b16.yaml
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = (
        parser.parse_args()
    )  # 解析命令行传入的参数，并将它们存储在一个命名空间对象 args 中
    args = parser.parse_args()  # 解析命令行传入的参数
    main(args)
