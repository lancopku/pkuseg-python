# pkuseg：一个多领域中文分词工具包

pkuseg简单易用，支持细分领域分词，有效提升了分词准确度。



## 目录

* [主要亮点](#主要亮点)
* [编译和安装](#编译和安装)
* [各类分词工具包的性能对比](#各类分词工具包的性能对比)
* [使用方式](#使用方式)
* [相关论文](#相关论文)
* [作者](#作者)
* [常见问题及解答](#常见问题及解答)



## 主要亮点

pkuseg具有如下几个特点：

1. 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络文本领域和混合领域的分词预训练模型，同时也拟在近期推出更多的细领域预训练模型，比如医药、旅游、专利、小说等等。
2. 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。



## 编译和安装

- 目前**仅支持python3**
- **新版本发布：2019-1-23**
- **新版本特性：**
  - **修改了词典处理方法，扩充了词典，分词效果有提升**
  - **效率进行了优化，测试速度较之前版本提升9倍左右**
  - **增加了在大规模混合数据集训练的通用模型，并将其设为默认使用模型**
- **新版本发布：2019-1-30**
- **新版本特性：**
  - **支持fine-tune训练（从预加载的模型继续训练），支持设定训练轮数**
- **为了获得好的效果和速度，强烈建议大家通过pip install更新到目前的最新版本**

1. 通过PyPI安装(自带模型文件)：
	```
	pip3 install pkuseg
	之后通过import pkuseg来引用
	```
   **建议更新到最新版本**以获得更好的开箱体验(新版默认提供通用的预训练模型、默认开启词典)：
   	```
	pip3 install -U pkuseg
	```
2. 如果PyPI官方源下载速度不理想，建议使用镜像源，比如：   
   初次安装：
	```
	pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pkuseg
	```
   更新：
	```
	pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pkuseg
	```
3. 从GitHub下载(需要下载模型文件，见[预训练模型](#预训练模型))
	```
	将pkuseg文件放到目录下，通过import pkuseg使用
	模型需要下载或自己训练。
	```
	

## 各类分词工具包的性能对比

我们选择jieba、THULAC等国内代表分词工具包与pkuseg做性能比较。

考虑到jieba分词和THULAC工具包等并没有提供细领域的预训练模型，为了便于比较，我们重新使用它们提供的训练接口在细领域的数据集上进行训练，用训练得到的模型进行中文分词。

我们选择Linux作为测试环境，在新闻数据(MSRA)、混合型文本(CTB8)、网络文本(WEIBO)数据上对不同工具包进行了准确率测试。我们使用了第二届国际汉语分词评测比赛提供的分词评价脚本。其中MSRA与WEIBO使用标准训练集测试集划分，CTB8采用随机划分。对于不同的分词工具包，训练测试数据的划分都是一致的；**即所有的分词工具包都在相同的训练集上训练，在相同的测试集上测试**。对于需要训练的模型，如THULAC和pkuseg，在所有数据集上，我们使用默认的训练超参数。以下是pkuseg训练代码示例:

```
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models', nthread=20)
pkuseg.test('msr_test.raw', 'output.txt', user_dict=None)
```





#### 细领域训练及测试结果

以下是在不同数据集上的对比结果：

| MSRA   | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.01 |  89.88 |     88.42 |
| THULAC |     95.60 |  95.91 |     95.71 |
| pkuseg |     96.94 |  96.81 | **96.88** |


| CTB8   | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     88.63 |  85.71 |     87.14 |
| THULAC |     93.90 |  95.30 |     94.56 |
| pkuseg |     95.99 |  95.39 | **95.69** |

| WEIBO  | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.79 |  87.54 |     87.66 |
| THULAC |     93.40 |  92.40 |     92.87 |
| pkuseg |     93.78 |  94.65 | **94.21** |



#### 跨领域测试结果

我们选用了混合领域的CTB8语料的训练集进行训练，同时在其它领域进行测试，以模拟模型在“黑盒数据”上的分词效果。选择CTB8语料的原因是，CTB8属于混合语料，理想情况下的效果会更好；而且在测试中我们发现在CTB8上训练的模型，所有工具包跨领域测试都可以获得更高的平均效果。以下是跨领域测试的结果：

| CTB8 Training | MSRA  | CTB8  | PKU   | WEIBO | All Average | OOD Average |
| ------------- | ----- | ----- | ----- | ----- | ----------- | ----------- |
| jieba         | 82.75 | 87.14 | 87.12 | 85.68 | 85.67       | 85.18       |
| THULAC        | 83.50 | 94.56 | 89.13 | 91.00 | 89.55       | 87.88       |
| pkuseg        | 83.67 | 95.69 | 89.67 | 91.19 | 90.06       | **88.18**   |

其中，`All Average`显示的是在所有测试集(包括CTB8测试集)上F-score的平均，`OOD Average` (Out-of-domain Average)显示的是在除CTB8外其它测试集结果的平均。



#### 默认模型在不同领域的测试效果

考虑到很多用户在尝试分词工具的时候，大多数时候会使用工具包自带模型测试。为了直接对比“初始”性能，我们也比较了各个工具包的默认模型在不同领域的测试效果(感谢 @yangbisheng2009 的建议)。请注意，这样的比较只是为了说明默认情况下的效果，并不是公平的。

| Default | MSRA  | CTB8  | PKU   | WEIBO | All Average |
| ------- | :---: | :---: | :---: | :---: | :---------: |
| jieba  | 81.45 | 79.58 | 81.83 | 83.56 | 81.61       |
| THULAC |	85.55 | 87.84 | 92.29 | 86.65 | 88.08 |
| pkuseg | 88.24 | 88.61 | 89.88 | 90.64 | **89.34**   |

其中，`All Average`显示的是在所有测试集上F-score的平均。



## 使用方式

#### 代码示例

以下代码示例适用于python交互式环境。

代码示例1：使用默认配置进行分词（默认模型，默认词典）。
```python3
import pkuseg

seg = pkuseg.pkuseg()                                  # 以默认配置加载模型
text = seg.cut('我爱北京天安门')                        # 进行分词
print(text)
```

代码示例2：使用默认模型，并使用自定义词典。
```python3
import pkuseg

seg = pkuseg.pkuseg(user_dict='my_dict.txt')		# 加载默认模型，给定用户词典为当前目录下的"my_dict.txt"
text = seg.cut('我爱北京天安门')                         # 进行分词
print(text)
```

代码示例3：使用其它模型，不使用词典
```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='./ctb8', user_dict=None)     # 假设用户已经下载好了ctb8的模型并放在了'./ctb8'目录下，通过设置model_name加载该模型
text = seg.cut('我爱北京天安门')                              # 进行分词
print(text)
```

代码示例4：对文件分词（默认模型，默认词典）
```python3
import pkuseg

# 对input.txt的文件分词输出到output.txt中
# 使用默认模型，使用词典，开20个进程
pkuseg.test('input.txt', 'output.txt', nthread=20)     
```


代码示例5：训练新模型 （模型随机初始化）
```python3
import pkuseg

# 训练文件为'msr_training.utf8'
# 测试文件为'msr_test_gold.utf8'
# 训练好的模型存到'./models'目录下
# 训练模式下会保存最后一轮模型作为最终模型
# 目前仅支持utf-8编码，训练集和测试集要求所有单词以单个或多个空格分开
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')	
```


代码示例6：fine-tune训练（从预加载的模型继续训练）
```python3
import pkuseg

# 训练文件为'train.txt'
# 测试文件为'test.txt'
# 加载'./pretrained'目录下的模型，训练好的模型保存在'./models'，训练10轮
pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
```

#### 多进程分词

当将以上代码示例置于文件中运行时，如涉及多进程功能，请务必使用`if __name__ == '__main__'`保护全局语句，如：  
mp.py文件
```python3
import pkuseg

if __name__ == '__main__':
    pkuseg.test('input.txt', 'output.txt', nthread=20)
    pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models', nthread=20)	
```
运行
```
python3 mp.py
```
详见[无法使用多进程分词和训练功能，提示RuntimeError和BrokenPipeError](https://github.com/lancopku/pkuseg-python/wiki#3-无法使用多进程分词和训练功能提示runtimeerror和brokenpipeerror)。

**在Windows平台上，请当文件足够大时再使用多进程分词功能**，详见[关于多进程速度问题](https://github.com/lancopku/pkuseg-python/wiki#9-关于多进程速度问题)。

#### 参数说明

模型配置
```
pkuseg.pkuseg(model_name="default", user_dict="default")
	model_name		模型路径。默认是"default"表示我们预训练好的模型(仅对pip下载的用户)。
	                        用户可以填自己下载或训练的模型所在的路径如model_name='./models'。
	user_dict		设置用户词典。默认使用我们提供的词典。用户可以填自己的用户词典的路径，词典格式为一行一个词。填None表示不使用词典。
```

对文件进行分词
```
pkuseg.test(readFile, outputFile, model_name="default", user_dict="default", nthread=10)
	readFile		输入文件路径
	outputFile		输出文件路径
	model_name		模型路径。同pkuseg.pkuseg
	user_dict		设置用户词典。同pkuseg.pkuseg
	nthread			测试时开的进程数
```

模型训练
```
pkuseg.train(trainFile, testFile, savedir, train_iter=20, init_model=None)
	trainFile		训练文件路径
	testFile		测试文件路径
	savedir			训练模型的保存路径
	train_iter		训练轮数
	init_model		初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'
```



## 预训练模型

分词模式下，用户需要加载预训练好的模型。我们提供了三种在不同类型数据上训练得到的模型，根据具体需要，用户可以选择不同的预训练模型。以下是对预训练模型的说明：

- MSRA: 在MSRA（新闻语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1twci0QVBeWXUg06dK47tiA)

- CTB8: 在CTB8（新闻文本及网络文本的混合型语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1DCjDOxB0HD2NmP9w1jm8MA)

- WEIBO: 在微博（网络文本语料）上训练的模型。[下载地址](https://pan.baidu.com/s/1QHoK2ahpZnNmX6X7Y9iCgQ)

- MixedModel: 混合数据集训练的通用模型。随pip包附带的是此模型。[下载地址](https://pan.baidu.com/s/1Ej2TFTOLp84FDIPoQMtsMg)


其中，MSRA数据由[第二届国际汉语分词评测比赛](http://sighan.cs.uchicago.edu/bakeoff2005/)提供，CTB8数据由[LDC](https://catalog.ldc.upenn.edu/ldc2013t21)提供，WEIBO数据由[NLPCC](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html)分词比赛提供。


欢迎更多用户可以分享自己训练好的细分领域模型。



## 版本历史

- v0.0.11(2019-01-09)
  - 修订默认配置：CTB8作为默认模型，不使用词典
- v0.0.14(2019-01-23)
  - 修改了词典处理方法，扩充了词典，分词效果有提升
  - 效率进行了优化，测试速度较之前版本提升9倍左右
  - 增加了在大规模混合数据集训练的通用模型，并将其设为默认使用模型
- v0.0.15(2019-01-30)
  - 支持fine-tune训练（从预加载的模型继续训练），支持设定训练轮数


## 开源协议
1. 本代码采用MIT许可证。
2. 欢迎对该工具包提出任何宝贵意见和建议，请发邮件至jingjingxu@pku.edu.cn。



## 相关论文

该代码包主要基于以下科研论文，如使用了本工具，请引用以下论文：
* Xu Sun, Houfeng Wang, Wenjie Li. Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection. ACL. 253–262. 2012 

```
@inproceedings{DBLP:conf/acl/SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea- Volume 1: Long Papers},
pages = {253--262},
year = {2012}}
```



## 常见问题及解答


1. [为什么要发布pkuseg？](https://github.com/lancopku/pkuseg-python/wiki#1-为什么要发布pkuseg)
2. [pkuseg使用了哪些技术？](https://github.com/lancopku/pkuseg-python/wiki#2-pkuseg使用了哪些技术)
3. [无法使用多进程分词和训练功能，提示RuntimeError和BrokenPipeError。](https://github.com/lancopku/pkuseg-python/wiki#3-无法使用多进程分词和训练功能提示runtimeerror和brokenpipeerror)
4. [是如何跟其它工具包在细领域数据上进行比较的？](https://github.com/lancopku/pkuseg-python/wiki#4-是如何跟其它工具包在细领域数据上进行比较的)
5. [在黑盒测试集上进行比较的话，效果如何？](https://github.com/lancopku/pkuseg-python/wiki#5-在黑盒测试集上进行比较的话效果如何)
6. [如果我不了解待分词语料的所属领域呢？](https://github.com/lancopku/pkuseg-python/wiki#6-如果我不了解待分词语料的所属领域呢)
7. [如何看待在一些特定样例上的分词结果？](https://github.com/lancopku/pkuseg-python/wiki#7-如何看待在一些特定样例上的分词结果)
8. [关于运行速度问题？](https://github.com/lancopku/pkuseg-python/wiki#8-关于运行速度问题)
9. [关于多进程速度问题？](https://github.com/lancopku/pkuseg-python/wiki#9-关于多进程速度问题)



## 作者

Ruixuan Luo （罗睿轩）,  Jingjing Xu（许晶晶）, Xuancheng Ren（任宣丞）, Yi Zhang（张艺）, Bingzhen Wei（位冰镇）， Xu Sun （孙栩）

