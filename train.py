import argparse
import torch
import os
import sys

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

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
import datasets.imagenet100

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.vpt
import trainers.promptsrc   
import pdb


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


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
    cfg.TRAINER.STRATEGY = "supervised"  # supervised, semi-supervised

    # 无标注数据的 shots
    cfg.DATASET.NUM_UNLABELED_SHOTS = 0

    cfg.TRAIN_OR_TEST = "train"
    cfg.INPUT.BASE_CONFIDENCE_BOUND = 0.9
    cfg.INPUT.NEW_CONFIDENCE_BOUND = 0.9


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 加载 yaml 配置文件并将其合并到此 CfgNode

    # 1. 从 dataset 的 config 中导入
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)  # cfg.DATASET.NAME

    # 2. 从使用的 method 相关的 config 中导入
    if args.config_file:
        cfg.merge_from_file(
            args.config_file
        )  # config/trainers/CoOp/vit_b16.yaml 下的内容

    # 3. 从输入的 args 中导入
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()  # 冻结 cfg

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("---------------------------------------------")
    print(cfg)
    print("---------------------------------------------")
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        sys.stdout.flush()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )  # 创建一个 ArgumentParser 对象，用来处理命令行输入
    parser.add_argument(
        "--root", type=str, default="", help="path to dataset"
    )  # /mnt/hdd/zhurui/data
    parser.add_argument(
        "--output-dir", type=str, default="", help="output directory"
    )  # output/caltech101/CoOp/vit_b16_1shots/nctx16_cscFalse_ctpend/seed1
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
    )  # configs/datasets/caltech101.yaml
    parser.add_argument(
        "--trainer", type=str, default="", help="name of trainer"
    )  # CoOp
    parser.add_argument(
        "--backbone", type=str, default="", help="name of CNN backbone"
    )  # vit_16
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
    main(args)
    os._exit(0)
