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
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from numpy import *


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
    for p in modelk.parameters():
        p.requires_grad = False
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


    clone =5
    internal = 6                       
    mus = Variable(torch.from_numpy(np.array([120])).float(), requires_grad=True)
    optim_mus = Adam([mus], lr=3.0)

    ratio = Variable(torch.from_numpy(np.concatenate([np.ones([3, ])])).float(), requires_grad=True)
    optim_ratio = Adam([ratio], lr=0.25)
    print('current mus is: ' + str(mus.item()))
    print('current ratio is: ' + str(ratio))
    # start training -----------------------------------------------------------
    for epoch in range(int(start_epoch),100):
        if epoch %2 != 0:
            print('epoch: '+str(epoch)+' let us learn the augmentation!')
            reward_list = []
            dist = MultivariateNormal(mus, torch.eye(1) * 100)
            sigma_list = dist.sample((clone,))
            print('sigma_list: '+str(sigma_list))

            # unfreeze
            if args.freeze == epoch and args.finetune is None:
                print('Unfreezing')
                for param in model.parameters():
                    param.requires_grad = True

            baseline_path  = './baseline_weight.pt'
            torch.save(model.state_dict(), baseline_path)
            for i in range(clone):
                model.load_state_dict(torch.load(baseline_path))
                print('loaded ' + baseline_path)
                for p in range(internal):
                    logs = train_one_epoch(i, model, modelk, train_loader, device, optimizer, P4_loss, epoch,
                        lr_scheduler, sigma_list[i], ratio, debug_mode=args.debug )
                acc_list = []
                for j in range(30, 255, 30):           #
                    acc = validate(j, model, modelk, valid_loader, device)
                    print('using para '+str(j)+' acc is '+str(acc))
                    acc_list.append(acc)
                peak = (acc_list.index(max(acc_list))+1)*30
                print('peak is ' + str(peak))
                reward = (1-(sigma_list[i]/250 - peak/250)**2)+ (1 - mean(acc_list)/100)
                print('reward is ' + str(reward))
                reward_list.append(reward)
            loss_mu = 0
            dist = MultivariateNormal(mus, torch.eye(1) * 100)
            for k in range(clone):
                loss_mu = loss_mu - dist.log_prob(sigma_list[k]) * (reward_list[k] - mean(reward_list)) / (np.std(reward_list) + np.finfo(np.float32).eps.item())
            print('loss_mu is ' + str(loss_mu.item()))
            optim_mus.zero_grad()
            loss_mu.backward()
            optim_mus.step()
            print('current mus is: '+str(mus.item()))
            id = reward_list.index(max(reward_list))
            path = './model' + str(id) + '.pt'
            model.load_state_dict(torch.load(path))
            print('final loaded ' + path)
            print('')

        if epoch % 2 == 0:
            print('epoch: '+str(epoch)+' let us learn the sampling!')
            reward_list = []
            dist = MultivariateNormal(ratio, torch.eye(3) * 0.5)
            ratio_list = dist.sample((clone,))
            print('ratio_list: '+str(ratio_list))

            if args.freeze == epoch and args.finetune is None:
                print('Unfreezing')
                for param in model.parameters():
                    param.requires_grad = True
            baseline_path  = './baseline_weight.pt'
            torch.save(model.state_dict(), baseline_path)
            for i in range(clone):
                model.load_state_dict(torch.load(baseline_path))
                print('loaded ' + baseline_path)
                for p in range(internal):
                    train_one_epoch(i, model, modelk, train_loader, device, optimizer, P4_loss, epoch,
                        lr_scheduler, mus, ratio_list[i], debug_mode=args.debug)

                acc = validate_sampling(model, modelk, valid_loader, device)
                print('acc is '+str(acc))

                reward = acc
                print('reward is ' + str(reward))
                reward_list.append(reward)
            loss_ratio = 0
            dist = MultivariateNormal(ratio, torch.eye(3) * 0.5)
            for k in range(clone):
                loss_ratio = loss_ratio - dist.log_prob(ratio_list[k]) * (reward_list[k] - mean(reward_list)) / (np.std(reward_list) + np.finfo(np.float32).eps.item())
            print('loss_ratio is ' + str(loss_ratio.item()))
            optim_ratio.zero_grad()
            loss_ratio.backward()
            optim_ratio.step()
            print('current ratio is: '+str(ratio))
            id = reward_list.index(max(reward_list))
            path = './model' + str(id) + '.pt'
            model.load_state_dict(torch.load(path))
            print('final loaded ' + path)
            print('')

    print("Training completed ")



def train_one_epoch(id,  model, modelk,train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler, sigma, ratio=None, debug_mode=False):
    training_start_time = time.time()
    lr_scheduler.step(epoch)
    samples_of_epoch = 0
    m=0
    # set model to train mode

    model.train()
    modelk.train()
    sigma = sigma.detach().numpy()
    noise = torch.tensor(np.random.randn(1, 1, 480, 640) * sigma / 255).cuda()

    ratio = ratio.int()
    for mode in range(3):
        if ratio[mode] > 3:
            ratio[mode] = 3
        elif ratio[mode] < 1:
            ratio[mode] = 1


    # summed loss of all resolutions
    total_loss_list = []

    for _, sample in enumerate(train_loader):


        start_time_for_one_step = time.time()

        # load the data and send them to gpu

        image = sample['image'].to(device)
        batch_size = image.data.shape[0]
        depth = sample['depth'].to(device)
        new = sample['new'].to(device)


        dropout = random.randint(1, 3)

        if dropout == 1: #3 modality

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

        elif dropout == 2: #2 modality
            imageq, depthq, newq = image.clone(), depth.clone(), new.clone()
            imageq = torch.zeros(imageq.shape).cuda()

            image_temp, depth_temp, new_temp = image.clone(), depth.clone(), new.clone()

            imagek2, depthk2, newk2 = imageq.clone(), depthq.clone(), newq.clone()

            depthk2 = depthk2[torch.randperm(depthk2.size(0))]

            depth = torch.zeros(depth.shape).cuda()
            new = torch.zeros(new.shape).cuda()

            image[:, 0, :, :] = image[:, 0, :, :] + noise
            image[:, 1, :, :] = image[:, 1, :, :] + noise
            image[:, 2, :, :] = image[:, 2, :, :] + noise

        elif dropout == 3: #2 modality

            depth = torch.zeros(depth.shape).cuda()
            new = torch.zeros(new.shape).cuda()

            imageq, depthq, newq = image.clone(), depth.clone(), new.clone()

            image[:, 0, :, :] = image[:, 0, :, :] + noise
            image[:, 1, :, :] = image[:, 1, :, :] + noise
            image[:, 2, :, :] = image[:, 2, :, :] + noise


        for param_q, param_k in zip(model.parameters(), modelk.parameters()):
            param_k.data = param_k.data.cuda() * m + param_q.data.cuda() * (1.00 - m)

        # optimizer.zero_grad()
        # this is more efficient than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        if dropout == 1:
            # forward pass
            _ , feat = model(image,depth,new)
            _ , featq = modelk(imageq,depthq,newq)
            _ , featk1 = modelk(imagek1,depthk1,newk1)
            _ , featk2 = modelk(imagek2,depthk2,newk2)
            _ , featk3 = modelk(imagek3,depthk3,newk3)

            # normalize
            feat = torch.nn.functional.normalize(feat.reshape(batch_size, -1), dim=1)
            featq = torch.nn.functional.normalize(featq.reshape(batch_size, -1), dim=1)
            featk1 = torch.nn.functional.normalize(featk1.reshape(batch_size, -1), dim=1)
            featk2 = torch.nn.functional.normalize(featk2.reshape(batch_size, -1), dim=1)
            featk3 = torch.nn.functional.normalize(featk3.reshape(batch_size, -1), dim=1)

            if ratio is not None:
                pool = [featk1, featk2, featk3]
                for m in range(3):
                    if ratio[m] > 1:
                        for n in range(ratio[m].item() - 1):
                            pool[m] = torch.cat((pool[m], pool[m]), 0)
                cat = torch.cat((featq, pool[0], pool[1], pool[2]), 0)
            else:
                cat = torch.cat((featq, featk1, featk2, featk3), 0)
            logits = torch.mm(feat, cat.transpose(1, 0))
            logits = logits.type(torch.float32)
            # print('logits: ' + str(logits))
            labels = torch.arange(feat.size()[0])
            labels = labels.cuda()
            loss = loss_function_train(logits/0.07, labels)
            # print('loss: '+str(loss.item()))

        elif dropout == 2:
            # forward pass
            _ , feat = model(image,depth,new)
            _ , featq = modelk(imageq,depthq,newq)
            _ , featk2 = modelk(imagek2,depthk2,newk2)

            # normalize            
            feat = torch.nn.functional.normalize(feat.reshape(batch_size , -1), dim=1)
            featq = torch.nn.functional.normalize(featq.reshape(batch_size , -1), dim=1)
            featk2 = torch.nn.functional.normalize(featk2.reshape(batch_size , -1), dim=1)
            cat = torch.cat((featq, featk2), 0)
            logits = torch.mm(feat, cat.transpose(1, 0))
            logits = logits.type(torch.float32)
            # print('logits: ' + str(logits))
            labels = torch.arange(feat.size()[0])
            labels = labels.cuda()
            loss = loss_function_train(logits/0.07, labels)
            # print('loss: '+str(loss.item()))

        elif dropout == 3:
            # forward pass
            _ , feat = model(image,depth,new)
            _ , featq = modelk(imageq,depthq,newq)

            # normalize
            feat = torch.nn.functional.normalize(feat.reshape(batch_size , -1), dim=1)
            featq = torch.nn.functional.normalize(featq.reshape(batch_size , -1), dim=1)
            logits = torch.mm(feat, featq.transpose(1, 0))
            logits = logits.type(torch.float32)
            # print('logits: ' + str(logits))
            labels = torch.arange(feat.size()[0])
            labels = labels.cuda()
            loss = loss_function_train(logits/0.07, labels)
            # print('loss: '+str(loss.item()))

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

        # print_log(epoch, samples_of_epoch, batch_size,
        #           len(train_loader.dataset), total_loss, time_inter,
        #           learning_rates)

        if debug_mode:
            # only one batch while debugging
            break

    path = './model'+str(id)+'.pt'
    torch.save(model.state_dict(), path)
    print('train done. '+path+' saved.')
    return 7

def validate(sigma, model, modelk, valid_loader, device):
    valid_split = valid_loader.dataset.split
    print(f'Validation on {valid_split}')

    for param_q, param_k in zip(model.parameters(), modelk.parameters()):
        param_k.data = param_q.data.cuda()

    noise = torch.tensor(np.random.randn(1, 1, 480, 640) * sigma / 255).cuda()
    # set model to eval mode
    model.eval()
    modelk.eval()

    acc_total = 0
    cnt = 0
    for i, sample in enumerate(valid_loader):
        # copy the data to gpu
        image = sample['image'].to(device)
        batch_size = image.shape[0]
        depth = sample['depth'].to(device)
        new = sample['new'].to(device)

        imageq, depthq, newq = image.clone(), depth.clone(), new.clone()
        imageq = torch.zeros(imageq.shape).cuda()

        depth = torch.zeros(depth.shape).cuda()
        new = torch.zeros(new.shape).cuda()

        image[:, 0, :, :] = image[:, 0, :, :] + noise
        image[:, 1, :, :] = image[:, 1, :, :] + noise
        image[:, 2, :, :] = image[:, 2, :, :] + noise

        # forward pass
        with torch.no_grad():

            prediction, feat = model(image, depth, new)
            _, feat = model(image, depth, new)
            _, featq = modelk(imageq, depthq, newq)

            feat = torch.nn.functional.normalize(feat.reshape(batch_size, -1), dim=1)
            featq = torch.nn.functional.normalize(featq.reshape(batch_size, -1), dim=1)
            logits = torch.mm(feat, featq.transpose(1, 0))
            logits = logits.type(torch.float32)
            labels = torch.arange(feat.size()[0])
            labels = labels.cuda()

            ind = torch.max(logits, 1)[1]
            dif = ind - labels
            sum = 0.0
            for i in range(batch_size):
                if dif[i] == 0:
                    sum = sum + 1
            acc = (sum / batch_size) * 100
            acc_total = acc_total + acc
            # print('acc: ' + str(acc))
            cnt = cnt + 1
    acc = acc_total*1.0 / cnt
    return acc

def validate_sampling(model, modelk, valid_loader, device):
    valid_split = valid_loader.dataset.split
    print(f'Validation on {valid_split}')

    for param_q, param_k in zip(model.parameters(), modelk.parameters()):
        param_k.data = param_q.data.cuda()

    # set model to eval mode
    model.eval()
    modelk.eval()

    acc_total = 0
    cnt = 0
    for i, sample in enumerate(valid_loader):
        # copy the data to gpu
        image = sample['image'].to(device)
        batch_size = image.shape[0]
        depth = sample['depth'].to(device)
        new = sample['new'].to(device)

        imageq, depthq, newq = image.clone(), depth.clone(), new.clone()

        # forward pass
        with torch.no_grad():
            choice = random.randint(1, 3)
            if choice == 1:
                _, feat = model(image, torch.zeros(depth.shape).cuda(),torch.zeros(new.shape).cuda())
                _, featq = modelk(torch.zeros(imageq.shape).cuda(), depthq, newq)
            elif choice == 2:
                _, feat = model(torch.zeros(image.shape).cuda(), depth, torch.zeros(new.shape).cuda())
                _, featq = modelk(imageq, torch.zeros(depthq.shape).cuda(), newq)
            elif choice == 3:
                _, feat = model(torch.zeros(image.shape).cuda(), torch.zeros(depth.shape).cuda(), new)
                _, featq = modelk(imageq, depthq, torch.zeros(newq.shape).cuda())

            feat = torch.nn.functional.normalize(feat.reshape(batch_size, -1), dim=1)
            featq = torch.nn.functional.normalize(featq.reshape(batch_size, -1), dim=1)
            logits = torch.mm(feat, featq.transpose(1, 0))
            logits = logits.type(torch.float32)
            labels = torch.arange(feat.size()[0])
            labels = labels.cuda()

            ind = torch.max(logits, 1)[1]
            dif = ind - labels
            sum = 0.0
            for i in range(batch_size):
                if dif[i] == 0:
                    sum = sum + 1
            acc = (sum / batch_size) * 100
            acc_total = acc_total + acc
            # print('acc: ' + str(acc))
            cnt = cnt + 1
    acc = acc_total*1.0 / cnt
    return acc


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
