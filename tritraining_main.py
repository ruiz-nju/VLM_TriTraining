import argparse
import sklearn.utils
import torch
import os
import sys

import random
from torchvision.transforms import InterpolationMode
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import copy
import numpy as np
import sklearn
from dassl.utils import read_image
from dassl.data.transforms.transforms import build_transform
from sklearn.metrics import accuracy_score
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns

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
import datasets.cifar100
import datasets.stl10
import datasets.imagenet100

import trainers.coop
import trainers.cocoop
import trainers.tri_training
import trainers.zsclip
import trainers.maple
import trainers.vpt
import trainers.promptsrc
import trainers.tcp
from trainers.tri_training import Tri_Training

import pdb


def setup_cfg(args, model_names):
    print("----------Build up cfg----------")
    prompts = ["a photo of a", "a photo of a", "a photo of a"]
    cfg = [get_cfg_default() for _ in range(3)]
    for i in range(3):
        extend_cfg(cfg[i])
        cfg[i].merge_from_file(args.dataset_config_file)
        cfg[i].merge_from_file(f"./configs/trainers/TriTraining/{model_names[i]}.yaml")
        cfg[i].TRAINER.PROMPTSRC.CTX_INIT = prompts[i]
        if args.root:
            cfg[i].DATASET.ROOT = args.root
        if args.output_dir:
            cfg[i].OUTPUT_DIR = osp.join(args.output_dir, "model_" + str(i))
        if args.model_dir:
            cfg[i].MODEL_DIR = osp.join(args.model_dir, "model_" + str(i))
        if args.resume:
            cfg[i].RESUME = args.resume
        if args.seed:
            cfg[i].SEED = args.seed
        if args.source_domains:
            cfg[i].DATASET.SOURCE_DOMAINS = args.source_domains
        if args.target_domains:
            cfg[i].DATASET.TARGET_DOMAINS = args.target_domains
        if args.transforms:
            cfg[i].INPUT.TRANSFORMS = args.transforms
        if args.backbone:
            cfg[i].MODEL.BACKBONE.NAME = args.backbone
        if args.head:
            cfg[i].MODEL.HEAD.NAME = args.head
        cfg[i].merge_from_list(args.opts)
        cfg[i].freeze()
    return cfg


def extend_cfg(cfg):
    cfg.TRAINER.MODAL = "classification"
    from yacs.config import CfgNode as CN

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.ALPHA = 1.0
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.COOP.W = 1.0
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
    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = (
        "a photo of a"  # initialization words (only for language prompts)
    )
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = (
        9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    )
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = (
        9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    )

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = (
        4  # number of context vectors at the vision branch
    )
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = (
        4  # number of context vectors at the language branch
    )
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = (
        9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    )
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = (
        9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    )
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1

    cfg.LOSS = CN()
    cfg.LOSS.GM = False
    cfg.LOSS.NAME = ""
    cfg.LOSS.ALPHA = 0.0
    cfg.LOSS.T = 1.0
    cfg.LOSS.LAMBDA = 1.0

    # 默认的采样设置为 all，可以根据需要进行调整
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # 训练策略可以根据需要进行调整
    cfg.TRAINER.STRATEGY = "semi-supervised"  # supervised, semi-supervised

    # 无标注数据的 shots
    cfg.DATASET.NUM_UNLABELED_SHOTS = 0

    cfg.TRAIN_OR_TEST = "train"
    cfg.INPUT.BASE_CONFIDENCE_BOUND = 0.9
    cfg.INPUT.NEW_CONFIDENCE_BOUND = 0.9
    cfg.INPUT.CONFIDENCE_BOUND = 0.9

def get_dataset(model):
    return (
        model.dm.dataset.train_x,
        model.dm.dataset.train_u,
        model.dm.dataset.val,
        model.dm.dataset.test,
    )


def main(args):
    model_names = [args.classifier, args.classifier, args.classifier]
    cfg = setup_cfg(args, model_names)

    base_cfg = cfg[0]
    setup_logger(args.output_dir)

    if torch.cuda.is_available() and base_cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    models = []
    models_tri = []
    warm_up_epochs = base_cfg.TRAIN.WARMUP
    train_x=None
    # Build up models
    if not args.eval_only:
        for i in range(3):
            print(f"----------Build up Base2new {model_names[i]}----------")
            set_random_seed(i + 1)
            # 只给base类别进行训练
            cfg[i].defrost()
            cfg[i].TRAINER.STRATEGY = "supervised"
            cfg[i].freeze()
            model = build_trainer(cfg[i])
            if i == 0:
                train_x, train_u, val, test = get_dataset(model)
                train_x = sklearn.utils.shuffle(train_x, random_state=base_cfg.SEED)
                print(f"train_x size: {len(train_x)}")
            sub_train_x = sklearn.utils.resample(train_x, replace=False, n_samples=len(train_x))
            print(f"Training model {i} for {warm_up_epochs} epochs")
            model.fit(sub_train_x, max_epoch=warm_up_epochs, Save=True)
    
    for i in range(3):
        print(f"----------Build up Tritrain {model_names[i]}----------")
        cfg[i].defrost()
        cfg[i].TRAINER.STRATEGY = "semi-supervised"
        cfg[i].freeze()
        model_tri = build_trainer(cfg[i])
        # print(cfg[i].MODEL_DIR)
        load_dirs = [osp.join(cfg[i].MODEL_DIR, model_names[i]) for i in range(3)]
        print(load_dirs[i])
        model_tri.custom_load_model(load_dirs[i], "warmup.pth.tar")
        models_tri.append(model_tri)

    train_x, train_u, val, test = get_dataset(models_tri[0])
    train_x = sklearn.utils.shuffle(train_x, random_state=base_cfg.SEED)
    train_u = sklearn.utils.shuffle(train_u, random_state=base_cfg.SEED)
    test_y = [datum.label for datum in test]

    if base_cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(base_cfg.SEED))
        set_random_seed(base_cfg.SEED)
    if args.eval_only:
        # output/TriTraining/base2novel_train/dtd/shots_16/unlabeled_shots_0/seed_1/model_0
        # PromptSRC
        load_dirs = [osp.join(cfg[i].MODEL_DIR, model_names[i]) for i in range(3)]
        for i in range(3):
            models_tri[i].custom_load_model(load_dirs[i])

        tri_trainer = Tri_Training(base_cfg, *models_tri)
        y_pred, y_pred_each_model = tri_trainer.predict(test)
        # 计算准确度
        accuracy = accuracy_score(test_y, y_pred)
        print(f"Accuracy: {accuracy}")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        plt.close()
        # acc_each_model = [accuracy_score(test_y, y) for y in y_pred_each_model]
        # for i in range(3):
        #     print(f"Accuracy of {model_names[i]}: {acc_each_model[i]}")
        sys.stdout.flush()
        return
    else:
        # 实例化 Tri_Training 并进行训练
        tri_trainer = Tri_Training(base_cfg, *models_tri)
        tri_trainer.fit(train_x, train_u, True)
        sys.stdout.flush()
        return


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
    )
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
        "--classifier", type=str, default="", help="name of classifier"
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
    os._exit(0)
