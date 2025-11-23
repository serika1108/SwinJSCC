import numpy as np
import torch
import random
import os
import logging
from torch.utils.tensorboard import SummaryWriter


class AverageMeter:
    """数值累加器"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(args, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        args.workdir += '_test'
    if save_log:
        # 创建日志目录
        makedirs(args.workdir)
        makedirs(args.samples)
        makedirs(args.save_models)

        # 配置日志
        formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
        stdhandler = logging.StreamHandler()
        stdhandler.setLevel(logging.INFO)
        stdhandler.setFormatter(formatter)
        logger.addHandler(stdhandler)
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

        # 添加可视化日志
        args.tb_logdir = os.path.join(args.workdir, "tb_logs")
        makedirs(args.tb_logdir)
        args.tb_writer = SummaryWriter(log_dir=args.tb_logdir)

    logger.setLevel(logging.INFO)
    args.logger = logger
    return args.logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
