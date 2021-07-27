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
from reid.evaluators import pairwise_distance
from reid.evaluators import reranking
from reid.evaluators import extract_features
from sklearn.cluster import DBSCAN
from reid.utils.data.sampler import RandomIdentitySampler
from reid.loss import TripletLoss

import os
os.environ["CUDA-VISIBLE_DEVICES"] = '0,1'
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
        shuffle=True, pin_memory=True, drop_last=True)
    
    target_train_loader = DataLoader(
        Preprocessor(dataset.target_train,
                     root=osp.join(dataset.target_images_dir, dataset.target_train_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

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
    dataset0, num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader = \
        get_data(args.data_dir, args.source, args.target, args.height,
                 args.width, args.batch_size, args.re, args.workers)


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, triplet_features=0)

    
    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    model = nn.DataParallel(model).cuda()
    
   

#############################################################################################

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
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        print(lr)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
            
            
            

    #Extract_features
    features, _= extract_features(model, target_train_loader, output_feature=args.output_feature)
    #computing distmat
     #distmat = pairwise_distance(features, features,dataset.target_train,dataset.target_train)
    distmat = reranking(features, features,dataset0.target_train,dataset0.target_train)
     #distmat=distmat.numpy()
    #clustering and labeling
    tri_mat = np.triu(distmat, 1) # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    top_num = np.round(args.rho*tri_mat.size).astype(int)
    eps0 = tri_mat[:top_num].mean()
    labels = DBSCAN(eps = eps0, min_samples = 4, metric='precomputed').fit_predict(distmat)

    #creat a new dataset 
    dataset=[]
    j=-1
    for i in features.items():
        j=j+1
        if labels[j] != -1:
              dataset.append((i[0],labels[j],0))


    file0 = open('cluster-duke-da-clt.txt', 'w')
    for i in range(len(dataset)):      
         file0.write(str(dataset[i]))
         file0.write('\n')
    file0.close()



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='msmt17',
                        choices=['market', 'duke', 'cuhk03_detected', 'msmt17'])
    # target
    parser.add_argument('-t', '--target', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17'])
    # images
    parser.add_argument('-b', '--batch_size', type=int, default=128, help="batch size for source")
    parser.add_argument('-b1', '--batchsize1', type=int, default=128)
    parser.add_argument('-b2', '--batchsize2', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)          
    parser.add_argument('-i','--iteration', type=int, default=0)    
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(triplet_batch_size // num_instances) identities, and "
                             "each identity has num_instances instances for source, "
                             "default: 8")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    #parser.add_argument('--resume', type=str, default='/home/zhx/GY/baseline/msmt17/checkpoint.pth.tar', metavar='PATH')
    parser.add_argument('--resume', type=str, default='./selftraining_duke/checkpoint.pth.tar', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--epochs_newtrain', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking

    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    main(parser.parse_args())
