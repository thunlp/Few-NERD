# %%
import random
from util.data_loader import Sample, get_class_name
from sklearn.model_selection import train_test_split
import os
import numpy as np
np.random.seed(0)
random.seed(0)

sample_modes = ['5-shot', '10%', '100%']
split_modes = ['supervised', 'intra', 'inter']

class ConllSample(Sample):
    def __init__(self, rawlines):
        self.lines = [[line.split(' ')[0], line.split(' ')[-1]] for line in rawlines]
        self.words, self.tags = zip(*self.lines)
        
class MySample(Sample):
    # no B-, I-
    # labelform: coarsetype-finegrainedtype
    def get_tag_coarse_class(self):
        # get coarse class name
        return list(set([tag.split('-')[0] for tag in self.tags if tag != 'O']))

    def __find_class__(self, tag, target_classes):
        for class_name in target_classes:
            if tag.startswith(class_name):
                return class_name
        return None

    def re_tag_coarse(self, target_classes):
        new_tags = []
        for tag in self.normalized_tags:
            class_name = self.__find_class__(tag, target_classes)
            if class_name:
                new_tags.append(tag)
            else:
                new_tags.append('O')
        self.new_tags = new_tags

    def re_tag(self, target_classes):
        new_tags = []
        for tag in self.normalized_tags:
            if tag in target_classes:
                new_tags.append(tag)
            else:
                new_tags.append('O')
        self.new_tags = new_tags


class DataSampler:
    def __init__(self, inputfile):
        self.class2sampleid = {}
        self.coarseclass2sampleid = {}
        self.samples = []
        self.inputfile = inputfile

    def __insert_sample__(self, d, index, sample_classes):
        for item in sample_classes:
            if item in d:
                d[item].append(index)
            else:
                d[item] = [index]

    def __write_to_file__(self, sample_list, file):
        print(file, len(sample_list))
        with open(file, 'w', encoding='utf-8')as f:
            f.writelines('\n\n'.join([str(sample) for sample in sample_list]))

    def __get_sample_idx_list__(self, d, class_list):
        indices = []
        for class_name in class_list:
            indices += d[class_name]
        return indices

    def __re_tag__(self, sample_list, target_classes, split_mode):
        if split_mode == 'intra':
            for sample in sample_list:
                sample.re_tag_coarse(target_classes)
        else:
            for sample in sample_list:
                sample.re_tag(target_classes)

    def load_conll(self):
        with open(self.inputfile, 'r')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('-DOCSTART-'):
                    continue
                samplelines.append(line)
            else:
                if samplelines:
                    sample = ConllSample(samplelines)
                    self.samples.append(sample)
                    sample_classes = sample.get_tag_class()
                    self.__insert_sample__(self.class2sampleid, index, sample_classes)
                    index += 1
                samplelines = []
    
    def load_my_data(self):
        with open(self.inputfile, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                if samplelines:
                    sample = MySample(samplelines)
                    self.samples.append(sample)
                    sample_classes = sample.get_tag_class()
                    coarse_sample_classes = sample.get_tag_coarse_class()
                    self.__insert_sample__(self.class2sampleid, index, sample_classes)
                    self.__insert_sample__(self.coarseclass2sampleid, index, coarse_sample_classes)
                    index += 1
                samplelines = []

    def split_data(self, split_mode, args):
        if split_mode not in split_modes:
            print('wrong split mode {}'.format(split_mode))
            return 
        if split_mode == 'supervised':
            trainval, test, _, _ = train_test_split(self.samples, [0]*len(self.samples), test_size=0.2, random_state=0)
            train, val, _, _ = train_test_split(trainval, [0]*len(trainval), test_size=0.125, random_state=0)
        else:
            if split_mode == 'intra':
                print(self.coarseclass2sampleid.keys())
                train_indices = self.__get_sample_idx_list__(self.coarseclass2sampleid, args['train-type'])
                val_indices = self.__get_sample_idx_list__(self.coarseclass2sampleid, args['val-type'])
                test_indices = self.__get_sample_idx_list__(self.coarseclass2sampleid, args['test-type'])
            else:
                if not args['train-type']:
                    print('randomly select type for inter mode')
                    train_type = []
                    val_type = []
                    test_type = []
                    # split finegrained types in each coarse type
                    for coarse_type in self.coarseclass2sampleid:
                        fine_types = [fine_type for fine_type in self.class2sampleid if fine_type.startswith(coarse_type)]
                        length = len(fine_types)
                        if length < 3:
                            print('not enough fine-grained types in [{}], val or test set may not contain the coarse type [{}]'.format(coarse_type, coarse_type))
                        permuted = np.random.permutation(length)
                        train_type += list(fine_types[i] for i in permuted[:(max(int(length * 0.6), 1))])
                        val_type += list(fine_types[i] for i in permuted[max(int(length * 0.6), 1):max(int(length * 0.8), 2)])
                        test_type += list(fine_types[i] for i in permuted[max(int(length * 0.8), 2):])
                        
                    args['train-type'] = train_type
                    args['val-type'] = val_type
                    args['test-type'] = test_type
                    print(train_type)
                    print(val_type)
                    print(test_type)
                train_indices = self.__get_sample_idx_list__(self.class2sampleid, args['train-type'])
                val_indices = self.__get_sample_idx_list__(self.class2sampleid, args['val-type'])
                test_indices = self.__get_sample_idx_list__(self.class2sampleid, args['test-type'])
            train_indices = list(set(train_indices))
            val_indices = list(set(val_indices).difference(set(train_indices)))
            test_indices = list(set(test_indices).difference(set(train_indices + val_indices)))
            #print(val_indices)
            train = [self.samples[i] for i in train_indices]
            val = [self.samples[i] for i in val_indices]
            test = [self.samples[i] for i in test_indices]
            #print(len(self.samples))
            #print(len(train))
            #print(len(val))
            #print(len(test))
            self.__re_tag__(train, args['train-type'], split_mode)
            self.__re_tag__(val, args['val-type'], split_mode)
            self.__re_tag__(test, args['test-type'], split_mode)
        self.__write_to_file__(train, args['train'])
        self.__write_to_file__(val, args['val'])
        self.__write_to_file__(test, args['test'])

            
    def process(self, sample_mode, outputfile):
        if sample_mode not in sample_modes:
            print('wrong sample mode {}'.format(sample_mode))
            return
        if sample_mode == '5-shot':
            sample_ids = []
            for class_name in self.class2sampleid:
                sample_ids += random.sample(self.class2sampleid[class_name], 5)
            sample_ids = list(set(sample_ids))
            selected_samples = [self.samples[i] for i in sample_ids]
        elif sample_mode == '10%':
            selected_samples = random.sample(self.samples, int(len(self.samples)*0.1))
        else:
            selected_samples = self.samples
        print(len(selected_samples))
        with open(outputfile, 'w', encoding='utf-8')as f:
            f.writelines('\n\n'.join([str(sample) for sample in selected_samples]))
# %%
if __name__ == '__main__':
    args = {'train': '', 'val': '', 'test': '', 'train-type': [], 'val-type': [],'test-type': []}
    args['train'] = 'data/mydata/train-intra-new.txt'
    args['val'] = 'data/mydata/val-intra-new.txt'
    args['test'] = 'data/mydata/test-intra-new.txt'
    args['train-type'] = ['person','other', 'art', 'product']
    args['val-type'] = ['event', 'building']
    args['test-type'] = ['organization','location']

    if not os.path.exists('data/mydata/'):
        os.mkdir('data/mydata/')
    sampler = DataSampler('data/processed_data_0131')
    sampler.load_my_data()
    sampler.split_data('intra', args)



# %%
