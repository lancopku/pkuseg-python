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

1. 高分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
2. 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。而其他现有分词工具包，一般仅提供通用领域模型。
3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。



## 编译和安装

1. 通过pip下载(自带模型文件)
	```
	pip install pkuseg
	之后通过import pkuseg来引用
	```
2. 从github下载(需要下载模型文件，见[预训练模型](#预训练模型))
	```
	将pkuseg文件放到目录下，通过import pkuseg使用
	模型需要下载或自己训练。
	```



## 各类分词工具包的性能对比

我们选择jieba、THULAC等国内代表分词工具包与pkuseg做性能比较。

考虑到jieba分词和THULAC工具包等并没有提供细领域的预训练模型，为了便于比较，我们重新使用它们提供的训练接口在细领域的数据集上进行训练，用训练得到的模型进行中文分词。

我们选择Linux作为测试环境，在新闻数据(MSRA)、混合型文本(CTB8)、网络文本(WEIBO)数据上对不同工具包进行了准确率测试。我们使用了第二届国际汉语分词评测比赛提供的分词评价脚本。其中MSRA与WEIBO使用标准训练集测试集划分，CTB8采用随机划分。对于不同的分词工具包，训练测试数据的划分都是一致的；**即所有的分词工具包都在相同的训练集上训练，在相同的测试集上测试**。以下是在不同数据集上的对比结果：


|MSRA | Precision | Recall | F-score|
|:------------|------------:|------------:|------------:|
| jieba |87.01 |89.88 |88.42 |
| THULAC | 95.60 | 95.91 | 95.71 |
| pkuseg | 96.94 | 96.81 | **96.88** |


|CTB8 | Precision | Recall | F-score|
|:------------|------------:|------------:|-------------:|
| jieba |88.63 |85.71 |87.14 |
| THULAC | 93.90 | 95.30 | 94.56 |
| pkuseg | 95.99 | 95.39 | **95.69** |

|WEIBO | Precision | Recall | F-score|
|:------------|------------:|------------:|-------------:|
| jieba |87.79 |87.54 |87.66 |
| THULAC | 93.40 | 92.40 | 92.87 |
| pkuseg | 93.78 | 94.65 | **94.21** |

同时，为了比较细领域分词的优势，我们比较了我们的方法和其通用分词模型的效果对比。其中jieba和THULAC均使用了软件包提供的、默认的分词模型：

|MSRA | F-score| Error Rate |
|:------------|------------:|------------:|
| jieba (Generic) |81.45 | 18.55 |
| THULAC (Generic) | 85.55 | 14.45 |
| pkuseg (Specific) | **96.88** | **3.12** |


|CTB8 | F-score | Error Rate|
|:------------|------------:|------------:|
|jieba (Generic)|79.58|20.42 |
|THULAC (Generic)|87.84|12.16 |
|pkuseg (Specific)| **95.69** |**4.31**|

从结果上来看，当用户了解待分词文本的领域时，细领域分词可以取得更好的效果。然而jieba和THULAC等分词工具包仅提供了通用领域模型。为了方便用户的使用和比较，我们预训练好的其它工具包的模型可以在[预训练模型](##预训练模型)节下载。




## 使用方式
1. 代码示例
	```
	代码示例1		使用默认模型及默认词典分词
	import pkuseg
	seg = pkuseg.pkuseg()				#以默认配置加载模型
	text = seg.cut('我爱北京天安门')	#进行分词
	print(text)
	```
	```
	代码示例2		设置用户自定义词典
	import pkuseg
	lexicon = ['北京大学', '北京天安门']	#希望分词时用户词典中的词固定不分开
	seg = pkuseg.pkuseg(user_dict=lexicon)	#加载模型，给定用户词典
	text = seg.cut('我爱北京天安门')		#进行分词
	print(text)
	```
	```
	代码示例3
	import pkuseg
	seg = pkuseg.pkuseg(model_name='./ctb8')	#假设用户已经下载好了ctb8的模型并放在了'./ctb8'目录下，通过设置model_name加载该模型
	text = seg.cut('我爱北京天安门')			#进行分词
	print(text)
	```
	```
	代码示例4
	import pkuseg
	pkuseg.test('input.txt', 'output.txt', nthread=20)	#对input.txt的文件分词输出到output.txt中，使用默认模型和词典，开20个进程
	```
	```
	代码示例5
	import pkuseg
	pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models', nthread=20)	#训练文件为'msr_training.utf8'，测试文件为'msr_test_gold.utf8'，模型存到'./models'目录下，开20个进程训练模型
	```
2. 参数说明
	```
	pkuseg.pkuseg(model_name='ctb8', user_dict=[])
	model_name		模型路径。默认是'ctb8'表示我们预训练好的模型(仅对pip下载的用户)。用户可以填自己下载或训练的模型所在的路径如model_name='./models'。
	user_dict		设置用户词典。默认不使用词典。填'safe_lexicon'表示我们提供的一个中文词典(仅pip)。用户可以传入一个包含若干自定义单词的迭代器。
	```
	```
	pkuseg.test(readFile, outputFile, model_name='ctb8', user_dict=[], nthread=10)
	readFile		输入文件路径
	outputFile		输出文件路径
	model_name		同pkuseg.pkuseg
	user_dict		同pkuseg.pkuseg
	nthread			测试时开的进程数
	```
	```
	pkuseg.train(trainFile, testFile, savedir, nthread=10)
	trainFile		训练文件路径
	testFile		测试文件路径
	savedir			训练模型的保存路径
	nthread			训练时开的进程数
	```



## 预训练模型

分词模式下，用户需要加载预训练好的模型。我们提供了三种在不同类型数据上训练得到的模型，根据具体需要，用户可以选择不同的预训练模型。以下是对预训练模型的说明：

- MSRA: 在MSRA（新闻语料）上训练的模型。新版本代码采用的是此模型。[下载地址](https://pan.baidu.com/s/1twci0QVBeWXUg06dK47tiA)

- CTB8: 在CTB8（新闻文本及网络文本的混合型语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1DCjDOxB0HD2NmP9w1jm8MA)

- WEIBO: 在微博（网络文本语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1QHoK2ahpZnNmX6X7Y9iCgQ)


其中，MSRA数据由[第二届国际汉语分词评测比赛](http://sighan.cs.uchicago.edu/bakeoff2005/)提供，CTB8数据由[LDC](https://catalog.ldc.upenn.edu/ldc2013t21)提供，WEIBO数据由[NLPCC](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html)分词比赛提供。



我们预训练好其它分词软件的模型可以在如下地址下载：

- jieba: To be uploaded
- THULAC: 在MSRA、CTB8、WEIBO、PKU语料上的预训练模型，[下载地址](https://pan.baidu.com/s/11L95ZZtRJdpMYEHNUtPWXA)，提取码：iv82

其中jieba的默认模型为统计模型，主要基于训练数据上的词频信息，我们在不同训练集上重新统计了词频信息。对于THULAC，我们使用其提供的接口进行训练(C++版本)，得到了在不同领域的预训练模型。

欢迎更多用户可以分享自己训练好的细分领域模型。




## 开源协议
1. pkuseg面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
2. 如有机构或个人拟将pkuseg用于商业目的，请发邮件至jingjingxu@pku.edu.cn洽谈技术许可协议。
3. 欢迎对该工具包提出任何宝贵意见和建议，请发邮件至jingjingxu@pku.edu.cn。



## 相关论文

本工具包基于以下文献：
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



## 作者

Ruixuan Luo （罗睿轩）,  Jingjing Xu（许晶晶）,  Xu Sun （孙栩）
