---
layout:     post           # 使用的布局（不需要改）
title:      BERT命名实体识别TensorFlow           # 标题 
subtitle:   BERT NER TensorFlow #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - 新能源
    - 命名实体识别
    - 专利


---

# BERT命名实体识别TensorFlow 

# 一、基于源码安装

```
git clone https://github.com/macanv/BERT-BiLSTM-CRF-NER
cd BERT-BiLSTM-CRF-NER/
python3 setup.py install
```

# 二、两个命令行工具：bert-base-ner-train 基于命名行训练命名实体识别模型

安装完bert-base后，会生成两个基于命名行的工具，其中bert-base-ner-train支持命名实体识别模型的训练，你只需要指定训练数据的目录，BERT相关参数的目录即可。可以使用下面的命令查看帮助

```
(tensorflow) root@e1fd7a8d1822:~/tian# bert-base-ner-train  -help
usage: bert-base-ner-train [-h] [-data_dir DATA_DIR]
                           [-bert_config_file BERT_CONFIG_FILE]
                           [-output_dir OUTPUT_DIR]
                           [-init_checkpoint INIT_CHECKPOINT]
                           [-vocab_file VOCAB_FILE]
                           [-max_seq_length MAX_SEQ_LENGTH] [-do_train]
                           [-do_eval] [-do_predict] [-batch_size BATCH_SIZE]
                           [-learning_rate LEARNING_RATE]
                           [-num_train_epochs NUM_TRAIN_EPOCHS]
                           [-dropout_rate DROPOUT_RATE] [-clip CLIP]
                           [-warmup_proportion WARMUP_PROPORTION]
                           [-lstm_size LSTM_SIZE] [-num_layers NUM_LAYERS]
                           [-cell CELL]
                           [-save_checkpoints_steps SAVE_CHECKPOINTS_STEPS]
                           [-save_summary_steps SAVE_SUMMARY_STEPS]
                           [-filter_adam_var FILTER_ADAM_VAR]
                           [-do_lower_case DO_LOWER_CASE] [-clean CLEAN]
                           [-device_map DEVICE_MAP] [-label_list LABEL_LIST]
                           [-verbose] [-ner NER] [-version]
bert-base-ner-train: error: argument -h/--help: ignored explicit argument 'elp'
```

训练的事例命名如下：

```
bert-base-ner-train \
    -data_dir {your dataset dir}\
    -output_dir {training output dir}\
    -init_checkpoint {Google BERT model dir}\
    -bert_config_file {bert_config.json under the Google BERT model dir} \
    -vocab_file {vocab.txt under the Google BERT model dir}
```

- data_dir是你的数据所在的目录，训练数据，验证数据和测试数据命名格式为：train.txt, dev.txt，test.txt,请按照这个格式命名文件，否则会报错。[数据标注代码参考](https://blog.csdn.net/u010189459/article/details/38546115?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
    训练数据的格式如下:

    ```
    海 O
    钓 O
    比 O
    赛 O
    地 O
    点 O
    在 O
    厦 B-LOC
    门 I-LOC
    与 O
    金 B-LOC
    门 I-LOC
    之 O
    间 O
    的 O
    海 O
    域 O
    。 O
    ```

    每行得第一个是字，第二个是它的标签，使用空格’ '分隔，请一定要使用空格。句与句之间使用空行划分。程序会自动读取你的数据。

-   output_dir： 训练模型输出的文件路径，模型的checkpoint以及一些标签映射表都会存储在这里，这个路径在作为服务的时候，可以指定为-ner_model_dir

-   init_checkpoint: 下载的谷歌BERT模型

-   bert_config_file ： 谷歌BERT模型下面的bert_config.json

-   vocab_file： 谷歌BERT模型下面的vocab.txt

    训练完成后，你可以在你指定的output_dir中查看训练结果。

# 三、bert-base-serving-start 将命名实体识别任务进行服务部署

先使用-help查看相关帮助

```
(tensorflow) root@e1fd7a8d1822:~/tian# bert-base-serving-start -help
usage: bert-base-serving-start [-h] -bert_model_dir BERT_MODEL_DIR -model_dir
                               MODEL_DIR [-model_pb_dir MODEL_PB_DIR]
                               [-tuned_model_dir TUNED_MODEL_DIR]
                               [-ckpt_name CKPT_NAME]
                               [-config_name CONFIG_NAME]
                               [-max_seq_len MAX_SEQ_LEN]
                               [-pooling_layer POOLING_LAYER [POOLING_LAYER ...]]
                               [-pooling_strategy {NONE,REDUCE_MAX,REDUCE_MEAN,REDUCE_MEAN_MAX,FIRST_TOKEN,LAST_TOKEN}]
                               [-mask_cls_sep] [-lstm_size LSTM_SIZE]
                               [-port PORT] [-port_out PORT_OUT]
                               [-http_port HTTP_PORT]
                               [-http_max_connect HTTP_MAX_CONNECT]
                               [-cors CORS] [-num_worker NUM_WORKER]
                               [-max_batch_size MAX_BATCH_SIZE]
                               [-priority_batch_size PRIORITY_BATCH_SIZE]
                               [-cpu] [-xla] [-fp16]
                               [-gpu_memory_fraction GPU_MEMORY_FRACTION]
                               [-device_map DEVICE_MAP [DEVICE_MAP ...]]
                               [-prefetch_size PREFETCH_SIZE] [-verbose]
                               [-mode MODE] [-version]
bert-base-serving-start: error: argument -h/--help: ignored explicit argument 'elp'
```

作为命名实体识别任务的服务，这两个目录是你必须指定的：ner_model_dir, bert_model_dir
然后你就可以使用下面的命令启动了：

```
bert-base-serving-start \
    -model_dir C:\workspace\python\BERT_Base\output\ner2 \
    -bert_model_dir F:\chinese_L-12_H-768_A-12
    -mode NER
```

-   bert_model_dir: 谷歌BERT模型的解压路径,可以在这里下载 https://github.com/google-research/bert
-   model_dir: 训练好的NER模型或者文本分类模型的路径，对于上面的output_dir
-   model_pd_dir: 运行模型优化代码后， 经过模型压缩后的存储路径，例如运行上面的命令后改路径下会产生 ner_model.pb 这个二进制文件
-   mode:NER 或者是BERT这两个模式，类型是字符串，如果是NER,那么就会启动NER的服务，如果是BERT，那么具体参数将和[bert as service] 项目中得一样。

# 四、在本地连接服务端进行命名实体识别的测试

......

# 五、自己训练命名实体识别模型

## 1. 下载Google BERT 预训练模型：

```
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
unzip chinese_L-12_H-768_A-12.zip
```

## 2. 训练模型

训练之前先在项目目录中新建一个output文件夹，模型的输出，和结构都会保存在这个目录中`mkdir output`

237上训练的命令（GPU）

```
bert-base-ner-train \
	-data_dir /root/tian/BERT-BiLSTM-CRF-NER/data/ne_WIPO_NER \
    -output_dir /root/tian/BERT-BiLSTM-CRF-NER/output/ne_WIPO_NER \
    -init_checkpoint /root/tian/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_model.ckpt \
    -bert_config_file /root/tian/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_config.json \
    -vocab_file /root/tian/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/vocab.txt \
    -save_summary_steps 20 \
    -batch_size 32 \
    -device_map 0
```

238上训练的命令（CPU）

```
bert-base-ner-train \
	-data_dir /root/tian/bert/BERT-BiLSTM-CRF-NER/data/ne_WIPO_NER \
    -output_dir /root/tian/bert/BERT-BiLSTM-CRF-NER/output/ne_WIPO_NER \
    -init_checkpoint /root/tian/bert/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_model.ckpt \
    -bert_config_file /root/tian/bert/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_config.json \
    -vocab_file /root/tian/bert/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/vocab.txt \
    -save_summary_steps 20
```







# Links:

1.  【新全】2019 NLP(自然语言处理)之Bert课程https://www.bilibili.com/video/av76791626?p=29
2.  基于BERT预训练的中文命名实体识别TensorFlow实现博客，主要参考的是这篇博客，还有GitHub库 https://blog.csdn.net/macanv/article/details/85684284    https://github.com/macanv/BERT-BiLSTM-CRF-NER      （根据这篇博客里训练完了模型后就可以进行预测了。python3 terminal_predict.py）
3.  Bert-BiLSTM-CRF-pytorch 一个GitHub库，看起来也不错 https://github.com/chenxiaoyouyou/Bert-BiLSTM-CRF-pytorch