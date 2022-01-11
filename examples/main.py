# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dualclustercontrast import datasets
from dualclustercontrast import models
from dualclustercontrast.trainers import DualClusterContrastTrainer
from dualclustercontrast.evaluators import Evaluator, extract_features
from dualclustercontrast.utils.data import IterLoader
from dualclustercontrast.utils.data import transforms as T
from dualclustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from dualclustercontrast.utils.data.preprocessor import Preprocessor
from dualclustercontrast.utils.logging import Logger
from dualclustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dualclustercontrast.utils.faiss_rerank import compute_jaccard_distance

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print(root)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances,  trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_val_loader(dataset, height, width, batch_size, workers, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    num_instances = 16
    train_set = trainset
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySampler(train_set, num_instances)
    test_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pooling_type=args.pooling_type)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


from torch.nn import Parameter
def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


import torch
import torch.nn.functional as F
from torch import nn, autograd
import random

class DCC(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut_ccc, lut_icc,  momentum):
        ctx.lut_ccc = lut_ccc
        ctx.lut_icc = lut_icc
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_ccc = inputs.mm(ctx.lut_ccc.t())
        outputs_icc = inputs.mm(ctx.lut_icc.t())

        return outputs_ccc,outputs_icc

    @staticmethod
    def backward(ctx, grad_outputs_ccc, grad_outputs_icc):
        inputs,targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs_ccc.mm(ctx.lut_ccc)+grad_outputs_icc.mm(ctx.lut_icc)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.data.cpu().numpy()):
            batch_centers[index].append(instance_feature)

        for y, features in batch_centers.items():
            mean_feature = torch.stack(batch_centers[y],dim=0)
            non_mean_feature = mean_feature.mean(0)
            x = F.normalize(non_mean_feature,dim=0)
            ctx.lut_ccc[y] = ctx.momentum * ctx.lut_ccc[y] + (1.-ctx.momentum) * x
            ctx.lut_ccc[y] /= ctx.lut_ccc[y].norm()

        del batch_centers 

        for x, y in zip(inputs,targets.data.cpu().numpy()):
            ctx.lut_icc[y] = ctx.lut_icc[y] * ctx.momentum + (1 - ctx.momentum) * x
            ctx.lut_icc[y] /= ctx.lut_icc[y].norm()

        return grad_inputs, None, None, None, None


def oim(inputs, targets, lut_ccc, lut_icc, momentum=0.1):
    return DCC.apply(inputs, targets, lut_ccc, lut_icc, torch.Tensor([momentum]).to(inputs.device))

import copy
class DCCLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True,init_feat=[]):
        super(DCCLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut_ccc', torch.zeros(num_classes, num_features).cuda())
        self.lut_ccc = copy.deepcopy(init_feat)

        self.register_buffer('lut_icc', torch.zeros(num_classes, num_features).cuda())
        self.lut_icc = copy.deepcopy(init_feat)

        print('Weight:{},Momentum:{}'.format(self.weight,self.momentum))

    def forward(self, inputs,  targets):
        inputs_ccc,inputs_icc = oim(inputs, targets, self.lut_ccc, self.lut_icc, momentum=self.momentum)

        inputs_ccc *= self.scalar
        inputs_icc *= self.scalar

        loss_ccc = F.cross_entropy(inputs_ccc, targets, size_average=self.size_average)
        loss_icc = F.cross_entropy(inputs_icc, targets, size_average=self.size_average)

        loss_con = F.smooth_l1_loss(inputs_ccc, inputs_icc.detach(), reduction='elementwise_mean')
        loss = loss_ccc+loss_icc+self.weight*loss_con

        return loss


import scipy.io as sio
def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    #iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    model.cuda()
    model = nn.DataParallel(model)
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    milestones=[50,100,150]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # Trainer
    trainer = DualClusterContrastTrainer(model)
    best_acc = 0.0
    for epoch in range(args.epochs):
        if epoch==0:
            with torch.no_grad():
                cluster_loader = get_test_loader(dataset, args.height, args.width,args.batch_size, args.workers, testset=sorted(dataset.train)) 
                features, labels = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            pseudo_labels = labels.data.cpu().numpy()
    
            cluster_features = generate_cluster_features(pseudo_labels, features)
            del cluster_loader, features

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            dcc_loss = DCCLoss(2048,num_cluster,weight= args.w, momentum = args.momentum, init_feat=F.normalize(cluster_features, dim=1).cuda())
            trainer.loss = dcc_loss

        train_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances)
        cluster_features = trainer.train(epoch, train_loader, optimizer,print_freq=args.print_freq)


        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} Moment:{}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, args.momentum,  mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual Cluster Contrastive Learning for person re-id")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.0,
                        help="update momentum for the hybrid memory")
    
    parser.add_argument('--w', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--step-size', type=int, default=50)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs_0914_adapt'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    main()
