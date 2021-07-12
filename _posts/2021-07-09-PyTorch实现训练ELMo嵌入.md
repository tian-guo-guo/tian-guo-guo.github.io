---
layout:     post           # 使用的布局（不需要改）
title:      PyTorch实现训练ELMo嵌入
subtitle:   PyTorch实现训练ELMo嵌入 #副标题
date:       2021-07-09             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
   - nlp


---





# [PyTorch实现训练ELMo嵌入](https://mp.weixin.qq.com/s/ZhE4S7m7J0q3O93hVJTwgA)

语言模型嵌入（**E**mbeddings from **L**anguage **Mo**del,ELMo）是一种功能强大的上下文嵌入方法，广泛应用于各种自然语言处理任务。

ELMo等人开创了自然语言处理中语境词预训练的趋势。该技术仍然简单直观，允许自己轻松地添加到现有模型中。

在本文中，我们将讨论如何用我们自己的文本语料库从头开始训练ELMo嵌入，并解释它是如何工作的。我们将使用AllenNLP，这是一个基于PyTorch的NLP框架，它提供了许多开箱即用的最先进的模型。如果你只对使用预训练过的ELMo嵌入感兴趣，请跳到最后一节-在下游任务中使用ELMo。

* * *

### 目录

1.  从头开始训练ELMo

2.  理解代码

3.  在下游任务中使用ELMo

* * *

### 从头开始训练ELMo

在下一节中，我们将逐步完成训练，以了解ELMo是如何在AllenNLP中实现的。

#### 依赖项

首先，让我们安装allennlp-models。这个包包括在AllenNLP框架中实现的所有高级模型。安装此软件包还可以找到所需的PyTorch和AllenNLP的版本。

    pip install allennlp-models=v2.0.1

#### 语料库

接下来，我们获取语料库数据进行训练。为了演示的目的，我们使用了原始语料库（http://www.statmt.org/lm-benchmark/）。你也可以使用任何你选择的语料库。

    wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
    tar -xzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
    export BIDIRECTIONAL_LM_DATA_PATH=$PWD'/1-billion-word-language-modeling-benchmark-r13output'
    export BIDIRECTIONAL_LM_TRAIN_PATH=$BIDIRECTIONAL_LM_DATA_PATH'/training-monolingual.tokenized.shuffled/*'

#### 词汇

为了索引和提高性能，AllenNLP需要预处理的词汇表，该词汇表从tokens.txt读取，其中每行代表一个唯一标记。

在这里，我们下载预处理的词汇表。如果你正在使用你自己的语料库，看看tokens.txt了解它的格式。请特别注意前三个标记</s>, <s>,@@UNKNOWN@@。它们对于ELMo了解词汇表之外的句子和单词的边界至关重要。

    pip install awscli 
    
    mkdir vocabulary 
    export BIDIRECTIONAL_LM_VOCAB_PATH=$PWD'/vocabulary' 
    cd $BIDIRECTIONAL_LM_VOCAB_PATH 
    
    aws --no-sign-request s3 cp s3://allennlp/models/elmo/vocab-2016-09-10.txt . 
    
    cat vocab-2016-09-10.txt | sed 's/<UNK>/@@UNKNOWN@@/' > tokens.txt
    rm vocab-2016-09-10.txt 
    
    echo '*labels\n*tags' > non_padded_namespaces.txt

#### 训练

最后一步了。在这一步中，我们使用一个预编译的配置文件来启动训练。

    wget https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/lm/bidirectional_language_model.jsonnet
    allennlp train bidirectional_language_model.jsonnet --serialization-dir output_dir/

#### 不满意?

上述步骤大量借鉴了官方的allinnlp HowTo。如果你想调整底层架构，或者你想将自己的语料库处理成所需的格式，这看起来可能相当不清楚。

虽然官方提供了惊人的简单性，但是AllenNLP扫除了大量关于数据和架构的潜在假设。

在你等待训练完成的同时，我们将从AllenNLP框架的角度来看看ELMo是如何工作的。

* * *

### 理解代码

为了理解ELMo体系结构以及它是如何实现的，我们将更仔细地研究一下bidirectional\_language\_model.jsonnet，它本质上是一个配置文件，它在非常高的级别上指定了所有内容。它允许我们定义如何处理数据（标记化和嵌入）以及体系结构。

注意：在JSON对象中，我们看到许多对象的键为“type”。这是AllenNLP用于注册自定义类的语法。通过使用它的修饰符，我们将类公开给AllenNLP，使我们能够在配置中使用它们。

#### DatasetReader

DatasetReader把自然语言和自然数联系起来

![图片](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210709232515.png)

我们首先关心的是数据如何从磁盘上的自然语言文本流入模型。数据首先由一个标记器进行标记，在空格处拆分句子。然后，TokenIndexer将单个标识转换为数字。在许多情况下，将单词转换成词汇表中各自的id就足够了。因为我们使用字符级CNN来捕获子词信息，所以我们需要按每个字符另外转换单词。

    local BASE_READER = {  
       "type": "simple_language_modeling",  
       "tokenizer": {    
           "type": "just_spaces"  
       },  
       "token_indexers": {
           "tokens": {      
               "type": "single_id"    
           },    
           "token_characters": {      
               "type": "elmo_characters"    
           }  
       },    
       "max_sequence_length": 400,  
       "start_tokens": ["<S>"],  
       "end_tokens": ["</S>"],
    };

模型和DatasetReader之间的界限在于内部状态是否可学习。更具体地说，从标识到索引号的转换是一个固定的过程，通常依赖于一个查找表，因此它属于DatasetReader。

#### 模型-embedders

该部分将索引连接到嵌入

![图片](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210709232747.png)

将这些索引号转换成非文本化的嵌入是一个可以学习的过程

注意token\_embedders如何与token\_indexers共享相同的key。这允许模型在内部与DatasetReader连接。

在ELMo中，我们只使用字符级的非文本化嵌入。因此，单词的索引被丢弃，字符的索引被输入到字符编码嵌入器。

![图片](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210709232810.png)

让我们仔细看看字符嵌入器。

在内部，它通过学习嵌入矩阵将字符索引转换为字符嵌入。然后，对于每个单词，它使用CNN通过卷积各个单词的字符嵌入来学习其单词嵌入。这样做给了我们一个非文本化的词嵌入，包括子单词信息。

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "empty"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": 512,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
    }

非文本化的词嵌入已经存在了很长一段时间。像word2vec、GloVe这样的嵌入一直是NLP的标准工作。

这种嵌入的一个长期问题是上下文的丢失。对于非文本化的嵌入，“bank”在“river bank”和“bank account”中的嵌入是相同的。然而，我们知道这些词有着截然不同的含义。

接下来，我们来看看ELMo如何用它的上下文化器处理这个问题。

#### 模型语境化

该部分实现通过看句子计算词嵌入

![图片](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210709232903.png)

在原著中，彼得斯等人使用biLSTM作为语境化工具。前向LSTM从前面的单词中获取上下文；后向LSTM从后面的单词中获取上下文。

例如，要将嵌入xₜ上下文化，前向LSTM查看x₁，…，xₜ₋₁并创建一个隐藏状态；后向LSTM查看xₜ₊₁，…，xₙ，并创建另一个隐藏状态。通过连接两个隐藏状态，我们得到了同时考虑前向上下文和后向上下文的上下文化嵌入。

在后来的一项研究中，Peters等人发现，虽然LSTM在下游任务中具有更高的准确性，但transformer在性能略有下降的情况下却具有更好的推理速度。

因此，我们实现使用了一个基于transformer的上下文化器。直观地说，它达到了LSTM试图达到的目标——在每个词嵌入中添加句子上下文。

    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1
    }

这就完成了ELMo体系结构，也就是说，我们在下游任务中使用生成的上下文嵌入。我们要讨论的最后一个部分是它的训练目标——双向语言模型损失。

#### 模型-损失

该部分用前面的词来预测_\_\__

给定一个N个标识序列（t₁，tУ，…，tₙ），前向语言模型通过对给定历史（t₁，…，tₖ₋）的标识概率建模来计算序列的概率

![图片](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210709232937.png)

![图片](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

类似地，我们可以导出向后的语言模型。我们最终的双向语言模型目标仅仅是语料库中所有句子的两个log概率之和。

在AllenNLP中，损失被写入到language\_model中，因此我们只需在配置中将参数bidirective设置为true。我们还可以通过设置num\_samples使用sampled softmax loss来加快训练速度。

    "model": {
      "type": "language_model",
      "bidirectional": true,
      "num_samples": 8192,
      "sparse_embeddings": false,
      "text_field_embedder": {
        // see above
      },
      "dropout": 0.1,
      "contextualizer": {
        // see above
      }
    }

呼！最后总结了ELMo体系结构及其在AllenNLP中的实现。当你阅读的时候，你的模型可能已经完成了训练（好的，可能没有）。在下一节中，我们将使用经过训练的ELMo来嵌入一些文本！

### 在下游任务中使用ELMo

现在你已经完成了训练，你应该能够在output\_dir/中看到每个epoch的度量，在那里你还可以找到经过训练的模型权重model.tar.gz。或者，你可以抓取一个预训练好的权重，并将其集成到你的模型中。

要将ELMo集成到AllenNLP模型配置中，我们只需将它用作另一个token\_embedder：

    "text_field_embedder": {  "token_embedders": {    "elmo": {      "type": "bidirectional_lm_token_embedder",      "archive_file": std.extVar('BIDIRECTIONAL_LM_ARCHIVE_PATH'),      "dropout": 0.2,      "bos_eos_tokens": ["<S>", "</S>"],      "remove_bos_eos": true,      "requires_grad": false    }  }},"text_field_embedder": {  "token_embedders": {    "elmo": {      "type": "elmo_token_embedder",      "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",      "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",      "do_layer_norm": false,      "dropout": 0.5    }  }},

不想使用配置文件？也可以在Python中直接调用ELMo：

```python
from allennlp_models.lm.modules.token_embedders.bidirectional_lm import BidirectionalLanguageModelTokenEmbedderfrom allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexerfrom allennlp.data.tokenizers.token_class import Tokenimport torchlm_model_file = "output_dir/model.tar.gz"sentence = "ELMo loves you"tokens = [Token(word) for word in sentence.split()]lm_embedder = BidirectionalLanguageModelTokenEmbedder(    archive_file=lm_model_file,    bos_eos_tokens=None)indexer = ELMoTokenCharactersIndexer()vocab = lm_embedder._lm.vocabcharacter_indices = indexer.tokens_to_indices(tokens, vocab)["elmo_tokens"]# batch大小为1indices_tensor = torch.LongTensor([character_indices])# 提取嵌入embeddings = lm_embedder(indices_tensor)[0]print(embeddings)
```

结束了！虽然AllenNLP的语法可能仍然让你感到困惑，但我们确实看到了它的强大功能，它可以用0行代码从头开始训练一个基于PyTorch的ELMo！

它还包括许多开箱即用的NLP模型。如果你对使用它们感兴趣，请查看它的官方指南和文档：https://docs.allennlp.org/v2.0.1/

* * *

**参考引用**

*   https://arxiv.org/abs/1802.05365

*   https://arxiv.org/abs/1808.08949

*   https://www.bioinf.jku.at/publications/older/2604.pdf

*   https://arxiv.org/abs/1706.03762

*   https://arxiv.org/abs/1803.07640

![图片](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  

