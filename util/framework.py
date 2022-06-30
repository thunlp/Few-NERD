import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import word_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from .viterbi import ViterbiDecoder


def get_abstract_transitions(train_fname, use_sampled_data=True):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    if use_sampled_data:
        samples = data_loader.FewShotNERDataset(train_fname, None, 1).samples
        tag_lists = []
        for sample in samples:
            tag_lists += sample['support']['label'] + sample['query']['label']
    else:
        samples = data_loader.FewShotNERDatasetWithRandomSampling(train_fname, None, 1, 1, 1, 1).samples
        tag_lists = [sample.tags for sample in samples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotNERModel(nn.Module):
    def __init__(self, my_word_encoder, ignore_index=-1):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.ignore_index = ignore_index
        self.word_encoder = nn.DataParallel(my_word_encoder)
        self.cost = nn.CrossEntropyLoss(ignore_index=ignore_index)

    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    
    def __delete_ignore_index(self, pred, label):
        pred = pred[label != self.ignore_index]
        label = label[label != self.ignore_index]
        assert pred.shape[0] == label.shape[0]
        return pred, label

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        pred, label = self.__delete_ignore_index(pred, label)
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __transform_label_to_tag__(self, pred, query):
        '''
        flatten labels and transform them to string tags
        '''
        pred_tag = []
        label_tag = []
        current_sent_idx = 0 # record sentence index in the batch data
        current_token_idx = 0 # record token index in the batch data
        assert len(query['sentence_num']) == len(query['label2tag'])
        # iterate by each query set
        for idx, num in enumerate(query['sentence_num']):
            true_label = torch.cat(query['label'][current_sent_idx:current_sent_idx+num], 0)
            # drop ignore index
            true_label = true_label[true_label!=self.ignore_index]
            
            true_label = true_label.cpu().numpy().tolist()
            set_token_length = len(true_label)
            # use the idx-th label2tag dict
            pred_tag += [query['label2tag'][idx][label] for label in pred[current_token_idx:current_token_idx + set_token_length]]
            label_tag += [query['label2tag'][idx][label] for label in true_label]
            # update sentence and token index
            current_sent_idx += num
            current_token_idx += set_token_length
        assert len(pred_tag) == len(label_tag)
        assert len(pred_tag) == len(pred)
        return pred_tag, label_tag

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span
                
    def metrics_by_entity(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        pred_class_span = self.__get_class_span_dict__(pred)
        label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def error_analysis(self, pred, label, query):
        '''
        return 
        token level false positive rate and false negative rate
        entity level within error and outer error 
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        fp = torch.sum(((pred > 0) & (label == 0)).type(torch.FloatTensor))
        fn = torch.sum(((pred == 0) & (label > 0)).type(torch.FloatTensor))
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        within, outer, total_span = self.__get_type_error__(pred, label, query)
        return fp, fn, len(pred), within, outer, total_span


class FewShotNERFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, viterbi=False, N=None, train_fname=None, tau=0.05, use_sampled_data=True):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.viterbi = viterbi
        if viterbi:
            abstract_transitions = get_abstract_transitions(train_fname, use_sampled_data=use_sampled_data)
            self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''
        print("Start training...")
    
        # Init optimizer
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        it = 0
        while it + 1 < train_iter:
            for _, (support, query) in enumerate(self.train_data_loader):
                label = torch.cat(query['label'], 0)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = label.cuda()

                logits, pred = model(support, query)
                assert logits.shape[0] == label.shape[0], print(logits.shape, label.shape)
                loss = model.loss(logits, label) / float(grad_iter)
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                iter_loss += self.item(loss.data)
                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall)
                    sys.stdout.write('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                sys.stdout.flush()

                if (it + 1) % val_step == 0:
                    _, _, f1, _, _, _, _ = self.eval(model, val_iter)
                    model.train()
                    if f1 > best_f1:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1)  == train_iter:
                    break
                it += 1
                
        print("\n####################\n")
        print("Finish training " + model_name)
    
    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            sent_scores = emissions_list[i].cpu()
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy().tolist()
            for label in vit_labels:
                pred.append(label-1)
        return torch.tensor(pred).cuda()

    def eval(self,
            model,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))

        with torch.no_grad():
            it = 0
            while it + 1 < eval_iter:
                for _, (support, query) in enumerate(eval_dataset):
                    label = torch.cat(query['label'], 0)
                    if torch.cuda.is_available():
                        for k in support:
                            if k != 'label' and k != 'sentence_num':
                                support[k] = support[k].cuda()
                                query[k] = query[k].cuda()
                        label = label.cuda()
                    logits, pred = model(support, query)
                    if self.viterbi:
                        pred = self.viterbi_decode(logits, query['label'])

                    tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct

                    fn_cnt += self.item(fn.data)
                    fp_cnt += self.item(fp.data)
                    total_token_cnt += token_cnt
                    outer_cnt += outer
                    within_cnt += within
                    total_span_cnt += total_span

                    if it + 1 == eval_iter:
                        break
                    it += 1

            epsilon = 1e-6
            precision = correct_cnt / (pred_cnt + epsilon)
            recall = correct_cnt / (label_cnt + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
            within_error = within_cnt / (total_span_cnt + epsilon)
            outer_error = outer_cnt / (total_span_cnt + epsilon)
            sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
            sys.stdout.flush()
            print("")
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error
