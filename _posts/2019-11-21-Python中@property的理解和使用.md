---
layout: post # 使用的布局（不需要改）
title: Python中@property的理解和使用 # 标题
subtitle: Python中@property的理解和使用 #副标题
date: 2019-11-21 # 时间
author: 甜果果 # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 学习日记
  - python
---

## 1. 起源

在绑定属性时，如果我们直接把属性暴露出去，虽然写起来很简单，但是，没办法检查参数，导致可以把成绩随便改：

```python
s = Student()
s.score = 9999
```

这显然不合逻辑。为了限制 score 的范围，可以通过一个 set_score()方法来设置成绩，再通过一个 get_score()来获取成绩，这样，在 set_score()方法里，就可以检查参数：

```python
class Student(object):

    def get_score(self):
        return self._score

    def set_score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

现在，对任意的 Student 实例进行操作，就不能随心所欲地设置 score 了：

```python
>>> s = Student()
>>> s.set_score(60) # ok!
>>> s.get_score()
60
>>> s.set_score(9999)
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```

但是，上面的调用方法又略显复杂，没有直接用属性这么直接简单。

有没有既能检查参数，又可以用类似属性这样简单的方式来访问类的变量呢？当然有！Python 内置的@property 装饰器就是负责把一个方法变成属性调用的。

## 2. 使用@property

为了方便,节省时间,我们不想写 s.set_score(9999)啊,直接写 s.score = 9999 不是更快么，于是对以上代码加以修改。

```python
class Student(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self,value):
        if not isinstance(value, int):
            raise ValueError('分数必须是整数才行呐')
        if value < 0 or value > 100:
            raise ValueError('分数必须0-100之间')
        self._score = value
```

现在再调用就比较方便

```python
>>> s = Student()
>>> s.score = 60 # OK，实际转化为s.set_score(60)
>>> s.score # OK，实际转化为s.get_score()
60
>>> s.score = 9999
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```

可以看到，把一个 get 方法变成属性，只需要加上@property 就可以了，此时，@property 本身又创建了另一个装饰器@score.setter，负责把一个 setter 方法变成属性赋值，这么做完后,我们调用起来既可控又方便。
