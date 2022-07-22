# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt, save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log
from PIL import ImageFilter
from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow

def parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()

    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(f'Adapting learning rate to {args.lr} because provided '
                      f'batch size differs from default batch size of 8.')

    return args

    # 0,01 vs  batchsize==8

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def train_main():

    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args, ckpt_dir)

    if args.valid_full_res:
        train_loader, valid_loader, valid_loader_full_res = data_loaders
    else:
        train_loader, valid_loader = data_loaders
        valid_loader_full_res = None

    n_classes_without_void = train_loader.dataset.n_classes_without_void

    # model building -----------------------------------------------------------
    model, device = build_model(args, n_classes=n_classes_without_void,modality=4)
    modelk, device = build_model(args, n_classes=n_classes_without_void,modality=4)
    # predictor = MLPHead(153600, 1024, 512)
    # predictor= predictor.cuda()
    if args.freeze > 0:
        print('Freeze everything but the output layer(s).')
        for name, param in model.named_parameters():
            if 'out' not in name:
                param.requires_grad = False

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to
    # train loss)

    P4_loss = torch.nn.CrossEntropyLoss()

    optimizer = get_optimizer(args, model)

    # in this script lr_scheduler.step() is only called once per epoch
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i['lr'] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0


    # start training -----------------------------------------------------------
    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        if args.freeze == epoch and args.finetune is None:
            print('Unfreezing')
            for param in model.parameters():
                param.requires_grad = True

        train_one_epoch(model, modelk, train_loader, device, optimizer, P4_loss, epoch,
            lr_scheduler, debug_mode=args.debug)

        torch.save(model.state_dict(), 'pretrained.pth')

    print("Training completed ")




def train_one_epoch(model, modelk,train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler , debug_mode=False):
    training_start_time = time.time()
    lr_scheduler.step(epoch)
    samples_of_epoch = 0
    m=0
    # set model to train mode

    model.train()
    modelk.train()
    sigma = 0
    noise = torch.tensor(np.random.randn(1, 1, 480, 640) * sigma/255).cuda()
    for p in modelk.parameters():
        p.requires_grad = False

    # summed loss of all resolutions
    total_loss_list = []

    for i, sample in enumerate(train_loader):

        start_time_for_one_step = time.time()

        # load the data and send them to gpu

        image = sample['image'].to(device)
        batch_size = image.data.shape[0]
        depth = sample['depth'].to(device)
        new = sample['new'].to(device)

        imageq, depthq, newq = image.clone(), depth.clone(), new.clone()
        image_temp, depth_temp, new_temp = image.clone(), depth.clone(), new.clone()
        imageq[:, 0, :, :] = imageq[:, 0, :, :] + noise
        imageq[:, 1, :, :] = imageq[:, 1, :, :] + noise
        imageq[:, 2, :, :] = imageq[:, 2, :, :] + noise

        imagek1, depthk1, newk1 = image.clone(), depth.clone(), new.clone()
        imagek2, depthk2, newk2 = image.clone(), depth.clone(), new.clone()
        imagek3, depthk3, newk3 = image.clone(), depth.clone(), new.clone()

        imagek1 = imagek1[torch.randperm(newk3.size(0))]
        depthk2 = depthk2[torch.randperm(newk3.size(0))]
        newk3 = newk3[torch.randperm(newk3.size(0))]




        for param_q, param_k in zip(model.parameters(), modelk.parameters()):
            param_k.data = param_k.data.cuda() * m + param_q.data.cuda() * (1.00 - m)

        # optimizer.zero_grad()
        # this is more efficient than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # forward pass
        _ , feat = model(image,depth,new)
        _ , featq = modelk(imageq,depthq,newq)
        _ , featk1 = modelk(imagek1,depthk1,newk1)
        _ , featk2 = modelk(imagek2,depthk2,newk2)
        _ , featk3 = modelk(imagek3,depthk3,newk3)

        # normalize
        feat = torch.nn.functional.normalize(feat.reshape(batch_size , -1), dim=1)
        featq = torch.nn.functional.normalize(featq.reshape(batch_size , -1), dim=1)
        featk1 = torch.nn.functional.normalize(featk1.reshape(batch_size , -1), dim=1)
        featk2 = torch.nn.functional.normalize(featk2.reshape(batch_size , -1), dim=1)
        featk3 = torch.nn.functional.normalize(featk3.reshape(batch_size , -1), dim=1)

        # loss
        cat = torch.cat((featq, featk1, featk2, featk3), 0)
        logits = torch.mm(feat, cat.transpose(1, 0))
        logits = logits.type(torch.float32)
        print('logits: ' + str(logits)) # comment this line
        labels = torch.arange(feat.size()[0])
        labels = labels.cuda()
        loss = loss_function_train(logits/0.07, labels)

        total_loss = loss

        total_loss.backward()
        optimizer.step()

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        total_loss = total_loss.cpu().detach().numpy()
        total_loss_list.append(total_loss)

        if np.isnan(total_loss):
            raise ValueError('Loss is None')

        # print log
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_lr()

        print_log(epoch, samples_of_epoch, batch_size,
                  len(train_loader.dataset), total_loss, time_inter,
                  learning_rates)

        if debug_mode:
            # only one batch while debugging
            break
    return 7


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(
            'Currently only SGD and Adam as optimizers are '
            'supported. Got {}'.format(args.optimizer))

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    train_main()
