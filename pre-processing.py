# -*- coding:utf-8 -*-
import json
import nltk
# 这里的下载要是不成功，就参考https://www.cnblogs.com/eksnew/p/12909814.html把数据包放到对象目录下，数据包就是文件夹里的tokenizer
# nltk.download('punkt')
# 分词工具
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

class DataItem:
    def __init__(self, item):
        self.item = item
        self.text = json.loads(item[0])['srcText']
        self.sentences = sent_tokenize(self.text)
        self.entity_value = self.__get_entities__()

    def __get_entities__(self):
        if json.loads(self.item[1]):
            label_bucket = json.loads(self.item[1])['ies']
        else:
            label_bucket = {}
        entity_value = []
        for entity in label_bucket:
            entity_word = entity['exact']
            coarse_type = entity['label']['value']
            fine_type  = entity['label']['children'][0]['value']
            pos_start = int(entity['positionStart'])
            pos_end = int(entity['positionEnd'])
            # 构建 entity_value = [词, 粗类，细类，位置]
            kv = [entity_word, coarse_type, fine_type, pos_start, pos_end]
            #kv[entity_word] = entity['labelValues'][0]
            entity_value.append(kv)
        # 对entity_value = [词, 粗类，细类，位置]进行排序，很重要，因为他们标注也不是按顺序标的
        entity_value.sort(key = lambda entity_value: entity_value[3])
        return entity_value

    def __minus_sent_pos__(self, kv, sent_pos):
        return kv[:3]+[kv[3]-sent_pos, kv[4]-sent_pos]

    def __split_sent_by_entities__(self, sent, sent_pos):
        length = len(sent)
        filtered_entities = [self.__minus_sent_pos__(kv, sent_pos) for kv in self.entity_value 
                            if kv[3] >= sent_pos and kv[4] < sent_pos + length]
        sent_parts = []
        index = 0
        for kv in filtered_entities:
            if kv[3] > index:
                sent_parts.append(sent[index:kv[3]])
            assert sent[kv[3]:kv[4]] == kv[0], print(sent[kv[3]:kv[4]], kv[0])
            sent_parts.append(kv[0])
            index = kv[4]
        if index < length:
            sent_parts.append(sent[index:])
        return sent_parts, filtered_entities

    def get_processed_data(self):
        processed_data = []
        for sent in self.sentences:
            # 找到句子开始位置
            sent_pos = self.text.find(sent)
            if sent_pos == -1:
                print('[Sentence Tokenize Error]:', sent)
                # 没有找到则跳过该句子
                break
            # 找到该句子所有entity，并按照entity切分句子
            sent_parts, filtered_entities = self.__split_sent_by_entities__(sent, sent_pos)
            index = 0
            for part in sent_parts:
                words = word_tokenize(part)
                if filtered_entities and part == filtered_entities[index][0]:
                    for word in words:
                        processed_data.append(word+'\t'+filtered_entities[index][1]+'-'+filtered_entities[index][2])
                    if index < len(filtered_entities)-1:
                        # 如果匹配成功，维护的filtered_entities的index也要+1
                        index += 1
                else:
                    for word in words:
                        # 似乎处理的是没毛病，你再检查一下，如果可以就先写成文件
                        processed_data.append(word+'\t'+'O')
            # 句子之间空一行
            processed_data.append('')
            assert index == max(0, len(filtered_entities)-1), print(self.text, filtered_entities)
        return '\n'.join(processed_data)

if __name__ == '__main__':
    root = '../'
    filepath_list = ['data_1', 'data_2', 'data_3', 'data_4', 'data_5']
    for filepath in filepath_list:
        output = []
        with open(root + filepath, 'r', encoding='utf-8') as reader:
            data = reader.readlines()
        for item in data:
            if item.strip():
                # '\001'是分段符号
                item = item.split('\001')
                dataitem = DataItem(item)
                output.append(dataitem.get_processed_data())
        with open('data/processed_{}'.format(filepath), 'w', encoding='utf-8')as f:
            f.writelines('\n'.join(output))
