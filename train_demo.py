from util.data_loader import get_loader
from util.framework import FewShotNERFramework
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from model.nnshot import NNShot
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inter',
            help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name, must be basic-bert, proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')
    parser.add_argument('--seed', type=int, default=0,
           help='random seed')


    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
           help='StructShot parameter to re-normalizes the transition probabilities')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length

    if opt.mode != 'supervised' and opt.model == 'basic-bert':
        print(f'[ERROR] {opt.model} cannot be run on {opt.mode} dataset!')
        return
    
    if opt.mode == 'supervised':
        print("Supervised NER")
    else:
        print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)
    print('loading model and tokenizer...')
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    word_encoder = BERTWordEncoder(
            pretrain_ckpt,
            max_length)

    print('loading data...')
    opt.train = f'data/{opt.mode}/train.txt'
    opt.test = f'data/{opt.mode}/test.txt'
    opt.dev = f'data/{opt.mode}/dev.txt'
    if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
        os.system(f'bash data/download.sh {opt.mode}')

    train_data_loader = get_loader(opt.train, word_encoder,
            N=trainN, K=K, Q=Q, batch_size=batch_size, max_length=max_length)
    val_data_loader = get_loader(opt.dev, word_encoder,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length)
    test_data_loader = get_loader(opt.test, word_encoder,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length)

        
    prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    if model_name == 'proto':
        print('use proto')
        model = Proto(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)
    elif model_name == 'nnshot':
        print('use nnshot')
        model = NNShot(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)
    elif model_name == 'structshot':
        print('use structshot')
        model = NNShot(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train, viterbi=True)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, fp16=opt.fp16,
                train_iter=opt.train_iter, warmup_step=int(opt.train_iter * 0.1), val_iter=opt.val_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    # test
    precision, recall, f1, fp, fn, within, outer = framework.eval(model, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp, fn, within, outer))

if __name__ == "__main__":
    main()
