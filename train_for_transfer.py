import os
import sys
import argparse
import time
from datetime import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_training_dataloader, get_test_dataloader, WarmUpLR

import data

CHECKPOINT_PATH = 'checkpoint'
LOG_DIR = 'runs'
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
SAVE_EPOCH = 100


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        writer.add_scalar('Train/loss', loss.item(), n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s LR: {:0.6f}'.format(epoch, finish - start, optimizer.param_groups[0]['lr']))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0


    for (images, labels) in test_dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()


        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_dataloader.dataset),
        correct.float() / len(test_dataloader.dataset),
        finish - start
    ))
    print()

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_dataloader.dataset), epoch)

    return correct.float() / len(test_dataloader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-num_classes', type=int, default=100, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-imgsize', type=int, default=232, help='input size')
    parser.add_argument('-dataset', type=str, default='cifar100', help='input size')
    parser.add_argument("-weight_decay", default=0.0, type=float)
    parser.add_argument("-label_smooth", default=0.0, type=float)
    parser.add_argument('-MCSH_margin', default=0.0, type=float)
    parser.add_argument('-pretrained', type=str, metavar='weight name')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    if args.net == 'resnet50':
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load('./pretrained/'+args.pretrained))
        net.fc = nn.Linear(2048, args.num_classes)
        if use_cuda:
            net = net.cuda()

    elif args.net == 'ViT_B':
        net = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False)
        net.load_state_dict(torch.load('./pretrained/'+args.pretrained))
        net.fc = nn.Linear(2048, args.num_classes)
        if use_cuda:
            net = net.cuda()
    else:
        print('Please select resnet50 or ViT_B')
        sys.exit()


    print('loading network -done')
 
    print('Training with: '+str(torch.cuda.device_count())+' gpus')
    net = nn.DataParallel(net) 

    EPOCH = 200
    MILESTONES = [100, 150, 180]

    train_dataloader = get_training_dataloader(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225),
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    test_dataloader = get_test_dataloader(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225),
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    if os.path.exists('./Test_Acc_pretrained_own_'+args.dataset+'.csv') == False:
        with open('./Test_Acc_pretrained_own_'+args.dataset+'.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['name', 'description', 'pretrained', 'top1 best acc', 'top1 last acc'])

    
    loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=2e-5)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    path_name = args.net
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.dataset, path_name, TIME_NOW)

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, args.net, TIME_NOW))


    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    for epoch in range(1, EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > 100 and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.module.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.module.state_dict(), weights_path)

    writer.close()


with open('./Test_Acc_pretrained_own_'+args.dataset+'.csv', 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args.net, args.pretrained, str(best_acc.item())[:7], str(acc.item())])
