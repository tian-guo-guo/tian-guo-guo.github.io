---
layout:     post           # 使用的布局（不需要改）
title:      skill-python
subtitle:   skill-python  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
 
---

# skill-python

## 1.matplotlib

![image-20201202105757861](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201202105758.png)

```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
# mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def read():
	name_list = ['BiLSTM-CRF','Glove-LSTM-CRF','BERT-BiLSTM-CRF']
	time1=[0.8499,0.9043,0.9211]
	time2=[0.8458,0.8972,0.9245]
	time3=[0.8479,0.9007,0.9228]
 
	location=np.arange(len(name_list))
	width=0.2
 
	plt.figure(figsize=(6,3))

	plt.bar(location, time1,width = width,label="准确率",alpha=0.8,color="w",edgecolor="k")
	for a,b in zip(location, time1):
		plt.text(a, b+0.005, "%.4f" % b, ha='center', va= 'bottom',fontsize=7)

	plt.bar(location+width, time2,tick_label = name_list,width = width,label="召回率",alpha=0.8,color="w",edgecolor="k",hatch=".....")
	for a,b in zip(location+width, time2):
		plt.text(a, b+0.005, "%.4f" % b, ha='center', va= 'bottom',fontsize=7)

	plt.bar(location+width*2, time3,width = width,label="F1值",alpha=0.8,color="w",edgecolor="k",hatch="/")
	for a,b in zip(location+width*2, time3):
		plt.text(a, b+0.005, "%.4f" % b, ha='center', va= 'bottom',fontsize=7)

	plt.title('')
	plt.ylim(0.8,0.94)
	plt.legend(loc=4)
	plt.savefig('./barchart.jpg', dpi=600)
	plt.show()  

if __name__ == '__main__':
	read()
```

