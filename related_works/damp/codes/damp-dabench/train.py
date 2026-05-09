import argparse
import warnings

import numpy as np
import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.optim import lr_scheduler as dassl_lr_scheduler
from dassl.utils import collect_env_info, set_random_seed, setup_logger

import trainers.damp  # noqa: F401

warnings.filterwarnings("ignore")

if not hasattr(np, "int"):
    np.int = int


def _patch_dassl_scheduler_for_torch210():
    base_cls = dassl_lr_scheduler._BaseWarmupScheduler
    if getattr(base_cls, "_damp_torch210_patched", False):
        return

    original_init = base_cls.__init__

    def _compat_init(self, optimizer, successor, warmup_epoch, last_epoch=-1, verbose=False):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        torch.optim.lr_scheduler._LRScheduler.__init__(self, optimizer, last_epoch)

    if torch.optim.lr_scheduler.LRScheduler.__init__.__code__.co_argcount == 3:
        base_cls.__init__ = _compat_init
        base_cls._damp_torch210_patched = True
        base_cls._damp_original_init = original_init


_patch_dassl_scheduler_for_torch210()


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    for key in sorted(args.__dict__):
        print(f"{key}: {args.__dict__[key]}")
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
    if args.seed >= 0:
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
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.MODEL.BACKBONE.LOCAL_PATH = ""
    cfg.MODEL.INIT_WEIGHTS_CTX = None
    cfg.MODEL.INIT_WEIGHTS_PRO = None

    cfg.DABENCH = CN()
    cfg.DABENCH.DATASET_NAME = ""
    cfg.DABENCH.NOTES = ""

    cfg.TRAINER.DAPL = CN()
    cfg.TRAINER.DAPL.N_DMX = 16
    cfg.TRAINER.DAPL.N_CTX = 16
    cfg.TRAINER.DAPL.CSC = False
    cfg.TRAINER.DAPL.PREC = "fp16"
    cfg.TRAINER.DAPL.T = 1.0
    cfg.TRAINER.DAPL.TAU = 0.5
    cfg.TRAINER.DAPL.U = 1.0

    cfg.TRAINER.DAMP = CN()
    cfg.TRAINER.DAMP.N_CTX = 16
    cfg.TRAINER.DAMP.N_CLS = 2
    cfg.TRAINER.DAMP.CSC = False
    cfg.TRAINER.DAMP.PREC = "fp16"
    cfg.TRAINER.DAMP.TAU = 0.5
    cfg.TRAINER.DAMP.U = 1.0
    cfg.TRAINER.DAMP.IND = 1.0
    cfg.TRAINER.DAMP.IM = 1.0
    cfg.TRAINER.DAMP.STRONG_TRANSFORMS = []

    cfg.OPTIM_C = cfg.OPTIM.clone()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print(f"** System info **\n{collect_env_info()}\n")

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="compatibility field; dabench resolves real dataset paths.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory for resume")
    parser.add_argument("--seed", type=int, default=-1, help="only non-negative values enable a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for adaptation")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for adaptation")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to method config")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to dataset config")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="directory holding saved checkpoints")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    main(parser.parse_args())
