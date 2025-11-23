import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_train_loader, get_test_loader
from utils import *
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
import math
import argparse
from loss.distortion import *
import time
import torchvision


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0", "False"}
    TRUTHY_STRINGS = {"on", "true", "1", "True"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    elif s in {True, False}:
        return s
    else:
        print(s)
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


parser = argparse.ArgumentParser(description='SwinJSCC')

# --------------------------------模型参数--------------------------------
parser.add_argument('--C', type=int, default='96', help='bottleneck dimension')
parser.add_argument('--downsample', default=4, type=int,
                    help='number of downsampling steps, but this number only holds true when the patch=2')
parser.add_argument('--patch_size', default=2, type=int, help='patch size of the transformer')
parser.add_argument('--window_size', default=8, type=int, help='window size of the swin transformer')
parser.add_argument('--mlp_ratio', default=4., type=float)
parser.add_argument('--qkv_bias', default=True, type=bool_flag,
                    help='disable qkv bias (default: enabled)')
parser.add_argument('--patch_norm', default=True, type=bool_flag,
                    help='disable qkv bias (default: enabled)')
parser.add_argument('--qk_scale', type=float, default=None, help='qk scale for attention')
parser.add_argument('--in_channel', default=3, type=int, help='the channels of input image')

parser.add_argument('--encoder_embed_dims', default='128,192,256,320', type=str,
                    help='encoder embedding dim of swin transformer in each layer')
parser.add_argument('--encoder_depth', default='2,2,6,2', type=str,
                    help='the encoder model depth of each layer in swin transformer')
parser.add_argument('--encoder_heads', default='4,6,8,10', type=str,
                    help='the encoder attention heads of swin transformer')

parser.add_argument('--decoder_embed_dims', default='320,256,192,128', type=str,
                    help='decoder embedding dim of swin transformer in each layer')
parser.add_argument('--decoder_depth', default='2,6,2,2', type=str,
                    help='the decoder model depth of each layer in swin transformer')
parser.add_argument('--decoder_heads', default='10,8,6,4', type=str,
                    help='the decoder attention heads of swin transformer')

parser.add_argument('--model_path', type=str, default=None,
                    help='the path of pretrained model for retrain or test')

# --------------------------------数据集--------------------------------
parser.add_argument('--train_data_dir', type=str, default='/home/serika/DIV2K',
                    help='training dataset path')
parser.add_argument('--test_data_dir', type=str, default='/home/serika/Kodak',
                    help='must be consistent with the testset')

# --------------------------------训练参数--------------------------------
parser.add_argument('--training', default=True, type=bool_flag, help='training or testing')
parser.add_argument('--distortion_metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (absolute lr)')
parser.add_argument('--tot_epoch', type=int, default=1500, help='total training epoches')
parser.add_argument('--save_model_freq', type=int, default=10, help='save model frequency')
parser.add_argument('--batch_size', default=8, type=int, help='batch size of the training set')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print_step', type=int, default=100, help='print frequency in log')
parser.add_argument('--log_save', default=True, type=bool_flag, help='training or testing')
parser.add_argument('--image_dims', type=int, default=256, help='image size during model initialization and training')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for optimizer')
parser.add_argument('--warmup_epochs', type=int, default=50, help='number of warmup epochs')
parser.add_argument('--hold_epochs', type=int, default=500, help='number of warmup epochs')


# --------------------------------信道参数--------------------------------
parser.add_argument('--channel_type', type=str, default='awgn', choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--multiple_snr', type=str, default='10', help='random, fixed or a list of snr value')
parser.add_argument('--pass_channel', default=False, type=bool_flag, help="use mask")
args = parser.parse_args()


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


def get_param_groups(net, weight_decay):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        # 常见做法：norm / bias 不加 weight decay
        if param.dim() == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def train_one_epoch(args, net, CalcuSSIM, train_loader, optimizer, epoch, logger):
    net.train()
    # 用列表创建了6个累加器AverageMeter实例
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]

    # 写入tensorboard
    writer = getattr(args, "tb_writer", None)
    num_batches = len(train_loader)

    for batch_idx, input in enumerate(train_loader):
        global_step = epoch * num_batches + batch_idx
        start_time = time.time()
        input = input.to(args.device)
        recon_image, CBR, SNR, mse, loss_G = net(input)
        loss = loss_G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)
        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)

        if (global_step + 1) % args.print_step == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {cur_lr}',
            ]))
            logger.info(log)
            if writer is not None:
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("train/psnr", psnrs.avg, global_step)
                writer.add_scalar("train/msssim", msssims.avg, global_step)
            for i in metrics:
                i.clear()
    for i in metrics:
        i.clear()


def test(args, net, logger):
    test_loader = get_test_loader(args)
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]

    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(args.device)

    # 每个信道环境和每个压缩率一一对应，得到一个2维矩阵的实验结果
    results_snr = np.zeros((len(args.multiple_snr)))
    results_cbr = np.zeros((len(args.multiple_snr)))
    results_psnr = np.zeros((len(args.multiple_snr)))
    results_msssim = np.zeros((len(args.multiple_snr)))
    run_time = np.zeros((len(args.multiple_snr)))

    save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join("./recon", save_time)
    os.makedirs(save_root, exist_ok=True)

    for i, SNR in enumerate(args.multiple_snr):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input, names = batch
                start_time = time.time()
                input = input.to(args.device)
                recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})'
                ]))
                logger.info(log)
                save_test_images(recon_image, names, batch_idx, SNR, args.C, save_root)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        run_time[i] = elapsed.avg
        for t in metrics:
            t.clear()

    logger.info("========== Test Summary ==========")
    logger.info("SNR: %s", results_snr.tolist())
    logger.info("CBR: %s", results_cbr.tolist())
    logger.info("PSNR: %s", results_psnr.tolist())
    logger.info("MS-SSIM: %s", results_msssim.tolist())
    logger.info("RUN TIME: %s", run_time.tolist())
    print("Finish Test!")


def save_test_images(recon_image, names, batch_idx, SNR, rate, save_root):
    batch_size = recon_image.size(0)
    for b in range(batch_size):
        img = recon_image[b].cpu()
        if isinstance(names[b], str):
            base_name = os.path.splitext(names[b])[0]
        else:
            # 如果 names 不是字符串，就用索引代替
            base_name = f"img_{batch_idx}_{b}"
        filename = f"{base_name}_snr{SNR}_rate{rate}_b{batch_idx}_i{b}.png"
        save_path = os.path.join(save_root, filename)
        torchvision.utils.save_image(img, save_path)


def build_scheduler(optimizer, warmup_epochs, hold_epochs, total_epochs):
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        if epoch < warmup_epochs + hold_epochs:
            return 1.0

        decay_epochs = total_epochs - warmup_epochs - hold_epochs
        if decay_epochs <= 0:
            return 1.0

        progress = float(epoch - warmup_epochs - hold_epochs) / float(decay_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


if __name__ == '__main__':
    seed_torch(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录日志
    args.filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.workdir = f'./history/{args.filename}'
    args.log = os.path.join(args.workdir, f'Log_{args.filename}.log')
    args.samples = os.path.join(args.workdir, 'samples')
    args.save_models = os.path.join(args.workdir, 'models')
    logger = logger_configuration(args, save_log=args.log_save)
    logger.info("Args: %s", args.__dict__)

    # 处理列表参数
    args.encoder_embed_dims = [int(x) for x in args.encoder_embed_dims.split(',')]
    args.encoder_depth = [int(x) for x in args.encoder_depth.split(',')]
    args.encoder_heads = [int(x) for x in args.encoder_heads.split(',')]

    args.decoder_embed_dims = [int(x) for x in args.decoder_embed_dims.split(',')]
    args.decoder_depth = [int(x) for x in args.decoder_depth.split(',')]
    args.decoder_heads = [int(x) for x in args.decoder_heads.split(',')]

    args.multiple_snr = [int(x) for x in args.multiple_snr.split(',')]

    # 创建网络
    net = SwinJSCC(args)
    net.to(args.device)

    if args.training:
        total_epochs = args.tot_epoch
        save_freq = args.save_model_freq
        train_loader = get_train_loader(args)
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(args.device)
        param_groups = get_param_groups(net, args.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=args.lr)
        scheduler = build_scheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            hold_epochs=args.hold_epochs,
            total_epochs=total_epochs,
        )

        for epoch in range(total_epochs):
            train_one_epoch(args, net, CalcuSSIM, train_loader, optimizer, epoch, logger)
            scheduler.step()
            if (epoch + 1) % save_freq == 0:
                save_path = os.path.join(args.save_models, f"EP{epoch + 1}.model")
                save_model(net, save_path=save_path)
                test(args, net, logger)
    else:
        if args.model_path:
            load_weights(net, args.model_path)
        test(args, net, logger)




