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
from reid.loss import LSRLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
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



#####################################################################################
    file0=open('cluster-duke-da-clt.txt','r')
    #file0=open('duke_cluster_dataset.txt','r')
    list0=file0.readlines()
    file0.close()
    cluster_dataset = []
    for name in list0:
     name=name.strip('\n')
     name=name[1:][:-1].split(',')
     cluster_dataset.append((name[0][1:][:-1],int(name[1]),int(name[2])))

    # find the number of ids
    classes = []
    for i in cluster_dataset:
     if i[1] not in classes:
      classes.append(i[1])
    number_classes = len(classes)    
#####################################################################################

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=number_classes, triplet_features=0)

    print(model)
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
    criterion = LSRLoss(epsilon=args.labelsmooth).cuda()
    criterion_t = TripletLoss(margin=0.3).cuda()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Schedule learning rate
    def adjust_lr(epoch):
        true_epoch = epoch + 1
        if epoch <= 10:
         lr = (3.5e-5)*(epoch/10) 
        if epoch > 10 and epoch <= 40:
         lr = 3.5e-4
        if epoch > 40 and epoch <= 70:
         lr = 3.5e-5
        if epoch > 70 :
         lr = 3.5e-6 
        print(lr)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
            
            
            

#####################################################################################        
        
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.RandomSizedRectCrop(256, 128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=args.re),
    ])
    print('##################')
    print(args.batch_size)     
  
    new_train_loader = DataLoader(
        Preprocessor(cluster_dataset, root='./data/duke/bounding_box_train',
                     transform=transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        sampler=RandomIdentitySampler(cluster_dataset, num_instances=4),
        pin_memory=True, drop_last=True)           
        
#######################################################################################        
        
    #training new dataloader
    for epoch in range(args.epochs_newtrain):
      print('epoch: ',epoch)
      #adjust_lr(start_epoch)
      adjust_lr(epoch)
      for i, inputs in enumerate(new_train_loader):
        
        imgs, _, pids, _=inputs

        imgs = imgs.cuda()
        pids = pids.cuda()
        outputs = model(imgs)
        #_,outputs = model(imgs)
        outputs_t = model(imgs,'pool5')
        loss_s = criterion(outputs, pids)
        loss_t, _ = criterion_t(outputs_t, pids)
        loss = loss_s + loss_t 
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
##############################################################################



##############################################################################  



##############################################################################  




################################################################################
  
    # iteration
    
    for iteration in range(args.iteration):
     
     print('clustering: ',iteration)   
     if iteration == 0:
      batchsize = args.batchsize1
     if iteration == 1:
      batchsize = args.batchsize2
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
     iterationdataset=[]
     j=-1
     for i in features.items():
        j=j+1
        if labels[j] != -1:
              iterationdataset.append((i[0],labels[j],0))



     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
     transformer = T.Compose([
        T.RandomSizedRectCrop(256, 128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=args.re),
     ])
     new_train_loader = DataLoader(
        Preprocessor(iterationdataset, root=osp.join(dataset0.target_images_dir, dataset0.target_train_path),
                     transform=transformer),
        batch_size=batchsize, num_workers=args.workers,
        sampler=RandomIdentitySampler(iterationdataset, num_instances=4),
        pin_memory=True, drop_last=True)
    #training new dataloader
     for epoch in range(args.epochs_newtrain):
      print('epoch: ',epoch)
      adjust_lr(start_epoch)
      for i, inputs in enumerate(new_train_loader):
        
        imgs, _, pids, _=inputs

        imgs = imgs.cuda()
        pids = pids.cuda()
        outputs = model(imgs)
        #_,outputs = model(imgs)
        loss, _ = criterion_t(outputs, pids)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    mAP,r1,r5,r10=evaluator.evaluate(query_loader, gallery_loader, dataset0.query, dataset0.gallery, args.output_feature, args.rerank)

   

    #log
    #file = open('new_result_softmax_lsr+triplet_adam_warmup_ls_duke0.txt','a');
    file = open('result-duke_lsr.txt','a');
    s = '\n'+'iteration'+' '+str(args.iteration)+'  '+'epoch:'+' '+str(args.epochs_newtrain)+'  '+'batchsize:'+str(args.batch_size)+'  '+'batchsize1:'+str(args.batchsize1)+'  '+'batchsize2:'+str(args.batchsize2)+'  '+'eraser:'+str(args.re)+' '+'lsr:'+str(args.labelsmooth)+'  '+'mAP:'+' '+str(mAP)[:6]+'  '+'r1:'+' '+str(r1)[:6]+'  '+'r5:'+' '+str(r5)[:6]+'  '+'r10:'+' '+str(r10)[:6]
    file.write(s);
    file.close();

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='msmt17',
                        choices=['market', 'duke', 'cuhk03_detected', 'msmt17'])
    # target
    parser.add_argument('-t', '--target', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17'])
    # images
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="batch size for source")
    parser.add_argument('-b1', '--batchsize1', type=int, default=64)
    parser.add_argument('-b2', '--batchsize2', type=int, default=64)
    parser.add_argument('-lsr', '--labelsmooth', type=float, default=0.15)
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
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    #parser.add_argument('--resume', type=str, default='/home/zhx/GY/baseline/msmt17/checkpoint.pth.tar', metavar='PATH')
    #parser.add_argument('--resume', type=str, default='/home/zhx/GY/baseline/logs/checkpoint.pth.tar', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--epochs_newtrain', type=int, default=120)
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
    parser.add_argument('--re', type=float, default=0.5)
    #  perform re-ranking

    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    main(parser.parse_args())
