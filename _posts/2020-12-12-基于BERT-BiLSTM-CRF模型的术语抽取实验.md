---
layout:     post           # 使用的布局（不需要改）
title:      基于BERT-BiLSTM-CRF模型的术语抽取实验
subtitle:   ne_MT实验 #副标题
date:       2020-12-12            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
   - nlp
   - 专利
   - 新能源
   - 术语


---

# 基于BERT-BiLSTM-CRF模型的术语抽取实验



# [GitHub地址及博客教程](https://blog.csdn.net/macanv/article/details/85684284)



# 论文原文

[新能源专利文本术语抽取研究](http://kns.cnki.net/kcms/detail/21.1106.TP.20210511.1556.002.html).小型微型计算机系统:1-10[2021-05-13].



# 一、数据集

B-TERM I-TERM O 标签，句与句之间空格隔开。

```
一 O
种 O
高 B-TERM
效 I-TERM
生 I-TERM
物 I-TERM
质 I-TERM
能 I-TERM
源 I-TERM
制 I-TERM
备 I-TERM
装 I-TERM
置 I-TERM
使 O
用 O
方 O
便 O

所 O
述 O
地 B-TERM
热 I-TERM
能 I-TERM
输 I-TERM
送 I-TERM
机 I-TERM
构 I-TERM
包 O
括 O
热 B-TERM
水 I-TERM
输 I-TERM
送 I-TERM
管 I-TERM
```

#  二、分词、标注、划分为训练集验证集和测试集

>   所有的文件地址记得与自己的对应上。

![image-20210112122948487](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210112122948.png)

```python
# -*- coding: utf-8 -*-
import codecs
import sys
import random
import jieba 

def cidian_quchong_paixu():
    terms = open('terms_plus.txt', 'r').readlines()
    terms_ok = open('terms.txt', 'w')
    terms = list(set(terms))
    terms.sort()
    for term in terms:
        terms_ok.write(term.strip())
        terms_ok.write('\n')

def fenci():
    jieba.load_userdict("terms.txt")
    lines = open('weapon.txt', 'r').readlines()
    lines_jieba = open('weapon_fenci.txt', 'w')
    for line in lines:
        res = ' '.join(jieba.cut(line, cut_all=False))
        lines_jieba.write(res)

def char_tagging(input_file, output_file):
    terms = [word.strip() for word in open('weapon_terms.txt', 'r').readlines()]
    terms = list(set(terms))
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if word in terms:
                if len(word) == 1:
                    output_data.write(word + " O\n")
                else:
                    output_data.write(word[0] + " B-TERM\n")
                    for w in word[1:len(word)-1]:
                        output_data.write(w + " I-TERM\n")
                    output_data.write(word[len(word)-1] + " I-TERM\n")
            else:
                for w in word:
                    output_data.write(w + " O\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()

base_destination = 'BERT4'
def prepare_3_files_BERT():
    ne_zh = open('weapon_fenci_BERT.txt', 'r').readlines()
    line = ''.join(ne_zh)
    ne_zh_list = line.split('\n\n')
    # print(len(ne_zh_list))
    np_number = [i for i in range(len(ne_zh_list))]
    np_shuffle = random.shuffle(np_number)
    # # 直接shuffle返回的是None，实际上不是返回，而是在原先的list上进行修改
    seven = len(ne_zh_list)//10*7
    two = len(ne_zh_list)//10*8
    # # print(seven)
    # # print(two)
    shuffle_train_num = np_number[:seven] # 23627
    shuffle_test_num = np_number[seven:two]
    shuffle_valid_num = np_number[two:]
    
    train_zh = open(base_destination + '/train.txt', 'w')
    test_zh = open(base_destination + '/test.txt', 'w')
    valid_zh = open(base_destination + '/dev.txt', 'w')
    
    train_data = [ne_zh_list[num] for num in shuffle_train_num]
    for line in train_data:
        word_list = line.split('\n')
        for word in word_list:
            train_zh.write(word.strip())
            train_zh.write('\n')
        train_zh.write('\n')

    test_data = [ne_zh_list[num] for num in shuffle_test_num]
    for line in test_data:
        word_list = line.split('\n')
        for word in word_list:
            test_zh.write(word.strip())
            test_zh.write('\n')
        test_zh.write('\n')

    valid_data = [ne_zh_list[num] for num in shuffle_valid_num]
    for line in valid_data:
        word_list = line.split('\n')
        for word in word_list:
            valid_zh.write(word.strip())
            valid_zh.write('\n')
        valid_zh.write('\n')
   
if __name__ == '__main__':
    cidian_quchong_paixu()
    fenci()
    char_tagging('weapon_fenci.txt', 'weapon_fenci_BERT.txt')
    prepare_3_files_BERT()
```



# 三、改下载下来的代码

teriminal_predict.py新建一个terminal_predict_term.py

![image-20210112123408681](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210112123408.png)

```python
# encoding=utf-8
# terminal_predict_term.py
"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""

import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
args = get_args_parser()

# import argparse 
# import sys 

# parser = argparse.ArgumentParser()
# parser.add_argument('infile', type=argparse.FileType('r'), nargs='?', default=sys.stdin)
# parser.add_argument('ofile', type=argparse.FileType('w'), nargs='?', default=sys.stdout)
# args_self = parser.parse_args()

model_dir = 'weapon/result/weapon_BERT4'
bert_dir = 'chinese_L-12_H-768_A-12'

is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        # line = ['本', '发', '明', '提', '供', '了', '一', '种', '风', '能', '混', '合', '动', '力', ...]
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # print(id2label)
        sentences = args.infile
        for sentence in sentences:
        # while True:
            # print('input the test sentence:')
            # sentence = str(input())
            # print(sentence)
            # 本发明提供了一种风能混合动力车及代步装置，主要涉及节能环保混合驱动技术领域。
            start = datetime.now()
            if len(sentence) < 2:
                # print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence) 
            # ['本', '发', '明', '提', '供', '了', '一', '种', '风', '能', '混', '合', '动', '力', ...]
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                        input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            # print(pred_label_result)
            #todo: 组合策略
            result = strage_combined_link_org_loc(sentence, pred_label_result[0])
            # print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result

def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        # print(', '.join(line))

    def write_terms_into_file(terms):
        terms_all = [i.word for i in terms]
        terms_only = list(set(terms_all))
        for term in terms_only:
            args.ofile.write(term.strip())
            args.ofile.write('\n')

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    terms = eval.get_result(tokens, tags)
    write_terms_into_file(terms)
    # print_output(terms, 'TERMS: ')


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)

class Result(object):
    def __init__(self, config):
        self.config = config
        self.terms = []

    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        # tokens = ['本', '实', '用', '新', '型', '还', '提', '供', '了', '该', '系', '统', '的', '控', ...]
        # tags = ['O', 'B', 'M', 'M', 'E', 'O', 'O', 'O', 'O', 'O', 'B', 'E', 'O', 'B', ...]
        self.result_to_json(tokens, tags)
        return self.terms

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        # string = ['本', '实', '用', '新', '型', '还', '提', '供', '了', '该', '系', '统', '的', '控', ...]
        # tags = ['O', 'B', 'M', 'M', 'E', 'O', 'O', 'O', 'O', 'O', 'B', 'E', 'O', 'B', ...]
        # BIO标记
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'TERM':
            self.terms.append(Pair(word, start, end, 'TERM'))


if __name__ == "__main__":
    predict_online()
```

2.  bert_base/train/train_helper.py

    第12行添加*import* sys 

    然后末尾几行改成

    ```python
        # add labels
        group2.add_argument('-label_list', type=str, default=None,
                            help='User define labels， can be a file with one label one line or a string using \',\' split')
    
        parser.add_argument('-verbose', action='store_true', default=False,
                            help='turn on tensorflow logging for debug')
        parser.add_argument('-ner', type=str, default='ner', help='which modle to train')
        parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    
        parser.add_argument('-infile', type=argparse.FileType('r'), nargs='?', default=sys.stdin)
        parser.add_argument('-ofile', type=argparse.FileType('w'), nargs='?', default=sys.stdout)
        return parser.parse_args()
    ```



# 四、基于BERT-BiLSTM-CRF的中文术语抽取模型

注意文件存放位置。

```bash
bert-base-ner-train \
	-data_dir weapon/data/BERT4 \
    -output_dir weapon/result/weapon_BERT4 \
    -init_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
    -bert_config_file chinese_L-12_H-768_A-12/bert_config.json \
    -vocab_file chinese_L-12_H-768_A-12/vocab.txt \
    -save_summary_steps 20 \
    -batch_size 32 \
    -device_map 2
```

应该会有这样的结果

```
processed 19960 tokens with 1519 phrases; found: 1528 phrases; correct: 1495.
accuracy:  99.61%; precision:  97.84%; recall:  98.42%; FB1:  98.13
             TERM: precision:  97.84%; recall:  98.42%; FB1:  98.13  1528
             
BERT4
accuracy:  99.48%; precision:  99.04%; recall:  99.21%; FB1:  99.13
             TERM: precision:  99.04%; recall:  99.21%; FB1:  99.13  6252
```



# 五、来一个新的句子识别里面的术语

记得更改terminal_predict_term.py里加载的模型。

```bash
CUDA_VISIBLE_DEVICES=4 python3 BERT-BiLSTM-CRF-NER/terminal_predict_term_weapon.py -infile BERT-BiLSTM-CRF-NER/weapon/data/origin4/predict_test.txt -ofile BERT-BiLSTM-CRF-NER/weapon/data/origin4/predict_test_term.txt
```

