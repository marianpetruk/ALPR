from __future__ import print_function, absolute_import

import os
import errno
import argparse
import time
# import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
# import torchvision.datasets as datasets

from tensorboardX import SummaryWriter

from .evaluation import accuracy, AverageMeter, final_preds, iou_loss, iou_metric
from .misc import save_checkpoint, save_pred, adjust_learning_rate
# from imutils import batch_with_heatmap
# from hourglass_net import HourglassNet
# from unet_seresnext import UNetSEResNext
# from fpn import FPNSeg
from .unet_resnet import UNetResNet
from .dataset import Cars

best_acc = 0
best_loss = 1000000.0


def main(args):
    global best_acc
    global best_loss

    # create checkpoint dir
    if not os.path.isdir(args.checkpoint):
        try:
            os.makedirs(args.checkpoint)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # create model
    model = UNetResNet(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    criterion = iou_loss

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    # optionally resume from a checkpoint
    title = 'keypoints_hg'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    dataset = Cars('data/cars_annotations.json', 'data/train')
    print("Dataset length", len(dataset))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        Cars('data/cars_annotations.json', 'data/train'),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Cars('data/cars_annotations.json', 'data/train', train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr

    experiment_name = "unet_rn101_bb_2points"
    save_path = "./logs/test_6k_{}".format(experiment_name)
    logger = SummaryWriter(save_path)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, logger, epoch, args.debug, args.flip)

        # evaluate on validation set
        valid_loss = validate(val_loader, model, criterion, args.num_classes, logger, epoch,
                              args.debug, args.flip)

        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('val_loss', valid_loss, epoch)

        # remember best acc and save checkpoint
        #         is_best = valid_acc > best_acc
        #         best_acc = max(valid_acc, best_acc)
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)


def train(train_loader, model, criterion, optimizer, logger, epoch, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # print(input_var.shape)
        # print(target_var.shape)

        # compute output
        output = model(input_var)
        score_map = output.data.cpu()

        # print("model output", output.shape)
        # print("target", target_var.shape)

        # print(target_var)

        loss = criterion(output, target_var)

        # print(loss.item())

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def validate(val_loader, model, criterion, num_classes, logger, epoch, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()

    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()

        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        score_map = output.data.cpu()

        if i == 0:
            disp_pred = inputs[0]
            for i in range(target.shape[1]):
                disp_pred -= score_map[0][i]
            logger.add_image('Prediction', disp_pred, epoch)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Keypoint Training')
    # Model structure
    parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=4, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=4, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=2, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    main(parser.parse_args())
