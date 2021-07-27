from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.domain_adaptation import DA
from reid import models
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.loss import TripletLoss
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.data.sampler_camera import RandomCameraSampler
import mmd


def get_data(data_dir, source, target, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),      
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(dataset.source_train, num_instances=4),
        pin_memory=True, drop_last=True)
        
        
    target_train_loader = DataLoader(
        Preprocessor(dataset.target_train, root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                     transform=train_transformer),
        batch_size=60, num_workers=workers,
        sampler=RandomCameraSampler(dataset.target_train, num_instances=10),
        pin_memory=True, drop_last=True)             

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader


def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader = \
        get_data(args.data_dir, args.source, args.target, args.height,
                 args.width, args.batch_size, args.re, args.workers)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_t = TripletLoss(margin=0.3).cuda()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
#######################################################################################        
        
    #training new dataloader
    mmd_loss = mmd.MMD_loss()
    for epoch in range(args.epochs):
      print('epoch: ',epoch)
      #adjust_lr(start_epoch)
      adjust_lr(epoch)
      for i, inputs_source in enumerate(source_train_loader):
      
        try:
         inputs_target = next(target_train_iter)
        except:
         target_train_iter = iter(target_train_loader)
         inputs_target = next(target_train_iter)      
 
        imgs_source, _, pids_source, _ = inputs_source
        imgs_source = imgs_source.cuda()
        pids_source = pids_source.cuda()
        imgs_target, _, _, _ = inputs_target
        imgs_target = imgs_target.cuda()

#########################################################################################



#########################################################################################        
        
        
        
        outputs_source_class = model(imgs_source)
        outputs_source_triplet = model(imgs_source,'pool5')
        outputs_target = model(imgs_target,'pool5')
        
        

        
        loss_source_class = criterion(outputs_source_class, pids_source)
        loss_source_triplet, _ = criterion_t(outputs_source_triplet, pids_source)
        loss_adapt = mmd_loss(outputs_source_triplet, outputs_target)
        loss_adapt_camera0 = mmd_loss(outputs_target, outputs_target[0:10])
        loss_adapt_camera1 = mmd_loss(outputs_target, outputs_target[10:20])
        loss_adapt_camera2 = mmd_loss(outputs_target, outputs_target[20:30])
        loss_adapt_camera3 = mmd_loss(outputs_target, outputs_target[30:40])
        loss_adapt_camera4 = mmd_loss(outputs_target, outputs_target[40:50])
        loss_adapt_camera5 = mmd_loss(outputs_target, outputs_target[50:60]) 
        loss_adapt_camera = loss_adapt_camera0+loss_adapt_camera1+loss_adapt_camera2+loss_adapt_camera3+loss_adapt_camera4+loss_adapt_camera5  
        a = 0.6 
        b = 1
        loss = a*loss_source_class + (1-a)*loss_source_triplet + b*(loss_adapt + loss_adapt_camera)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      if (epoch+1)%5 ==0 :
       save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
       }, fpath=osp.join('./baseline_market', 'checkpoint.pth.tar'))
   
##############################################################################

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='msmt17',
                        choices=['market', 'duke', 'cuhk03_detected', 'msmt17'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke', 'msmt17'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size for source")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'new_adapt_market_0_5'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
