import sys, os

import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# from models.resnet import *
import models.fle_resnet as resnet
import models.resnet
import models.fle_resnet

import logging
import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--model', type=str, required=True, choices=['resnet34', 'resnet18', 'resnet50'])
parser.add_argument('--method', type=str, default='resnet', choices=['resnet', 'fle_resnet'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--step_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--multistep', type=int, nargs='+', default=[30, 70])

args = parser.parse_args()
args.multistep = [int(i) for i in args.multistep]



logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"output/log-{args.method}-{args.mode}-{args.model}.txt", 'a')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s]   %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=4)
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total

# def add_importance_hist(writer, model:torch.Module):
#     model.parameters()


def train_model(model, train_loader, test_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.multistep, 0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.start_epoch, args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f, lr=%.7f" % (
                epoch, args.total_epochs, i, len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))
        model.eval()
        acc = eval(model, test_loader)
        logger.info("Epoch %d/%d, Acc=%.4f, lr=%.7f" % (epoch, args.total_epochs, acc, optimizer.param_groups[0]['lr']))
        if best_acc < acc:
            os.makedirs(f'checkpoints/{model.__class__.__name__}', exist_ok=True)
            torch.save(model, save_path)
            best_acc = acc
        scheduler.step()
    logger.info("Best Acc=%.4f" % (best_acc))


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

    # block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    block_prune_probs = [0.1] * 3 + [0.3] * 4 + [0.4] * 6 + [0.4] * 3  # 77.62%
    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1
        elif isinstance(m, resnet.Bottleneck):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            prune_conv(m.conv3, block_prune_probs[blk_id])
            blk_id += 1
    return model


def main():
    train_loader, test_loader = get_dataloader()
    logger.info(args)
    if args.mode == 'train':
        args.round = 0
        # model = resnet18(pretrained=False, num_classes=10)
        ckpt = f'checkpoints/ResNet/{args.method}-{args.model}-{args.mode}-round{args.round}.pth'
        model_zoo = getattr(models, args.method)
        if args.resume:
            model = torch.load(ckpt)
        else:
            model = getattr(model_zoo, args.model)(pretrained=False, num_classes=10)
        logger.info(model)
        train_model(model, train_loader, test_loader, ckpt)
    elif args.mode == 'prune':
        previous_ckpt = f'checkpoints/ResNet/{args.method}-{args.model}-{args.mode}-round{args.round}.pth'
        logger.info("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        #  model = getattr(resnet, args.model)(pretrained=False, num_classes=10)

        params0 = sum([np.prod(p.size()) for p in model.parameters()])
        prune_model(model)
        logger.info(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("Number of Parameters: %.1fM, prune rate %.4f" % (params / 1e6, (1 - params / params0)))
        train_model(model, train_loader, test_loader, previous_ckpt)
    elif args.mode == 'test':
        ckpt = f'checkpoints/ResNet/{args.method}-{args.model}-round{args.round}.pth'
        logger.info("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        logger.info(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        logger.info("Acc=%.4f\n" % (acc))


if __name__ == '__main__':
    main()
