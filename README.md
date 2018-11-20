# pkuseg-python：一个高准确度的中文分词工具包
pkuseg-python简单易用，支持多领域分词，在不同领域的数据上都大幅提高了分词的准确率。

## 目录
* [主要亮点](#主要亮点)
* [编译和安装](#编译和安装)
* [各类分词工具包的性能对比](#各类分词工具包的性能对比)
* [使用方式](#使用方式)
* [相关论文](#相关论文)
* [其它语言实现](#其它语言实现)
* [作者](#作者)

## 主要亮点

pkuseg是由北京大学语言计算与机器学习研究组研制推出的一套全新的中文分词工具包。pkuseg具有如下几个特点：

1. 高分词准确率。相比于其他的分词工具包，我们的工具包在不同领域的数据上都大幅提高了分词的准确度。根据我们的测试结果，pkuseg分别在示例数据集（MSRA和CTB8）上降低了79.33%和63.67%的分词错误率。
2. 多领域分词。我们训练了多种不同领域的分词模型。根据待分词的领域特点，用户可以自由地选择不同的模型。
3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。

## 编译和安装
1. 从github下载(需要下载模型文件，见[预训练模型](#预训练模型)，旧版本代码支持训练)
	```
	将pkuseg文件放到目录下，通过import pkuseg使用
	下载的模型需要放到pkuseg/models目录下(旧版本代码为oldversion/model目录)。
	```
2. 通过pip下载(自带模型文件，暂不支持训练)
	```
	pip install pkuseg
	之后通过import pkuseg来引用
	```

## 各类分词工具包的性能对比
我们选择THULAC、结巴分词等国内代表分词工具包与pkuseg做性能比较。我们选择Linux作为测试环境，在新闻数据(MSRA)和混合型文本(CTB8)数据上对不同工具包进行了准确率测试。我们使用了第二届国际汉语分词评测比赛提供的分词评价脚本。评测结果如下：


|MSRA | F-score| Error Rate |
|:------------|------------:|------------:|
| jieba |81.45 | 18.55
| THULAC | 85.48 |  14.52
| pkuseg | **96.75 (+13.18%)**| **3.25 (-77.62%)**


|CTB8 | F-score | Error Rate|
|:------------|------------:|------------:|
|jieba|79.58|20.42
|THULAC|87.77|12.23
|pkuseg| **95.64 (+8.97%)**|**4.36 (-64.35%)**


## 使用方式
1. 新版本代码(支持对给定字符串即时分词及对文件中的UTF8文本分词)
	```
	代码示例1
	import pkuseg
	seg = pkuseg.pkuseg()	#加载模型
	text = seg.cut('我爱北京天安门')	#进行分词
	print(text)
	```
	```
	代码示例2
	import pkuseg
	pkuseg.test('input.txt', 'output.txt')	#加载models中的模型，对input.txt的文件分词输出到output.txt中
	```
2. 旧版本代码(支持根据给定语料训练模型以及对文件中的UTF8文本分词)
	```
	cd oldversion
	分词模式：python3 main.py test [-input file] [-output file]
	训练模式：python3 main.py train [-train file] [-test file]
	从文本文件输入输出（注意均为UTF8文本）
	在config.py中有参数的设置，运行时根据运行环境的实际情况修改其中的nThread参数设置并行的进程数。
	```
	参数说明
	```
	test  分词
    train 训练
    [-input file] 用户指定的待分词文件
    [-output file] 用户指定的分词结果输出文件
    [-train file] & [-test file] 用户标注的语料，句子之间以换行符分隔，词语之间以空格分隔
    ```
	运行样例
	```
    python3 main.py test data/input.txt data/output.txt         分词
    python3 main.py train data/train.txt data/test.txt          根据指定的训练文件训练，训练模型会保存到./model目录下
	```


### 预训练模型
分词模式下，用户需要加载预训练好的模型。我们提供了三种在不同类型数据上训练得到的模型，根据具体需要，用户可以选择不同的预训练模型。以下是对预训练模型的说明：

MSRA: 在MSRA（新闻语料）上训练的模型。新版本代码采用的是此模型。[下载地址](https://pan.baidu.com/s/1twci0QVBeWXUg06dK47tiA)

CTB8: 在CTB8（新闻文本及网络文本的混合型语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1DCjDOxB0HD2NmP9w1jm8MA)

WEIBO: 在微博（网络文本语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1QHoK2ahpZnNmX6X7Y9iCgQ)

其中，MSRA数据由[第二届国际汉语分词评测比赛](http://sighan.cs.uchicago.edu/bakeoff2005/)提供，CTB8数据由[LDC](https://catalog.ldc.upenn.edu/ldc2013t21)提供，WEIBO数据由[NLPCC](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html)分词比赛提供。




## 开源协议
1. pkuseg面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
2. 如有机构或个人拟将pkuseg用于商业目的，请发邮件至xusun@pku.edu.cn洽谈技术许可协议。
3. 欢迎对该工具包提出任何宝贵意见和建议，请发邮件至jingjingxu@pku.edu.cn。

## 相关论文
若使用此工具包，请引用如下文章：
* Xu Sun, Houfeng Wang, Wenjie Li. Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection. ACL. 253–262. 2012 

```
@inproceedings{DBLP:conf/acl/SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea- Volume 1: Long Papers},
pages = {253--262},
year = {2012}}
```


* Jingjing Xu, Xu Sun. Dependency-based Gated Recursive Neural Network for Chinese Word Segmentation. ACL 2016: 567-572
```
@inproceedings{DBLP:conf/acl/XuS16,
author = {Jingjing Xu and Xu Sun},
title = {Dependency-based Gated Recursive Neural Network for Chinese Word Segmentation},
booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, {ACL} 2016, August 7-12, 2016, Berlin, Germany, Volume 2: Short Papers},
year = {2016}}
```

## 其它语言实现
pkuseg（C#版）：
https://github.com/lancopku/PKUSeg

## 作者

Ruixuan Luo （罗睿轩）,  Jingjing Xu（许晶晶）,  Xu Sun （孙栩）
