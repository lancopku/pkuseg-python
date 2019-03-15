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

1. 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，推荐使用在混合领域上训练的通用模型。各领域分词样例可参考 [**example.txt**](https://github.com/lancopku/pkuseg-python/blob/master/example.txt)。
2. 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。
4. 支持词性标注。


## 编译和安装

- 目前**仅支持python3**
- 新版本发布：2019-1-23
  - 修改了词典处理方法，扩充了词典，分词效果有提升
  - **效率进行了优化，测试速度较之前版本提升9倍左右**
  - 增加了在大规模混合数据集训练的通用模型，并将其设为默认使用模型
- 新版本发布：2019-1-30
  - 支持fine-tune训练（从预加载的模型继续训练），支持设定训练轮数
- 新版本发布：2019-2-20
  - **支持词性标注，增加了医疗、旅游细领域模型**
- **为了获得好的效果和速度，强烈建议大家通过pip install更新到目前的最新版本**

1. 通过PyPI安装(自带模型文件)：
	```
	pip3 install pkuseg
	之后通过import pkuseg来引用
	```
   **建议更新到最新版本**以获得更好的开箱体验：
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
   
3. 如果不使用pip安装方式，选择从GitHub下载，可运行以下命令安装：
	```
	python setup.py build_ext -i
	```
	
   GitHub的代码并不包括预训练模型，因此需要用户自行下载或训练模型，预训练模型可详见[release](https://github.com/lancopku/pkuseg-python/releases)。使用时需设定"model_name"为模型文件。

注意：**安装方式1和2目前仅支持linux(ubuntu)、mac、windows 64 位的python3版本**。如果非以上系统，请使用安装方式3进行本地编译安装。
	

## 各类分词工具包的性能对比

我们选择jieba、THULAC等国内代表分词工具包与pkuseg做性能比较。

考虑到jieba分词和THULAC工具包等并没有提供细领域的预训练模型，为了便于比较，我们重新使用它们提供的训练接口在细领域的数据集上进行训练，用训练得到的模型进行中文分词。

我们选择Linux作为测试环境，在新闻数据(MSRA)、混合型文本(CTB8)、网络文本(WEIBO)数据上对不同工具包进行了准确率测试。我们使用了第二届国际汉语分词评测比赛提供的分词评价脚本。其中MSRA与WEIBO使用标准训练集测试集划分，CTB8采用随机划分。对于不同的分词工具包，训练测试数据的划分都是一致的；**即所有的分词工具包都在相同的训练集上训练，在相同的测试集上测试**。对于所有数据集，pkuseg使用了不使用词典的训练和测试接口。以下是pkuseg训练和测试代码示例:

```
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')
pkuseg.test('msr_test.raw', 'output.txt', user_dict=None)
```





#### 细领域训练及测试结果

以下是在不同数据集上的对比结果：

| MSRA   | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.01 |  89.88 |     88.42 |
| THULAC |     95.60 |  95.91 |     95.71 |
| pkuseg |     96.94 |  96.81 | **96.88** |


| WEIBO  | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.79 |  87.54 |     87.66 |
| THULAC |     93.40 |  92.40 |     92.87 |
| pkuseg |     93.78 |  94.65 | **94.21** |




#### 默认模型在不同领域的测试效果

考虑到很多用户在尝试分词工具的时候，大多数时候会使用工具包自带模型测试。为了直接对比“初始”性能，我们也比较了各个工具包的默认模型在不同领域的测试效果。请注意，这样的比较只是为了说明默认情况下的效果，并不一定是公平的。

| Default | MSRA  | CTB8  | PKU   | WEIBO | All Average |
| ------- | :---: | :---: | :---: | :---: | :---------: |
| jieba  | 81.45 | 79.58 | 81.83 | 83.56 | 81.61       |
| THULAC |	85.55 | 87.84 | 92.29 | 86.65 | 88.08 |
| pkuseg | 87.29 | 91.77 | 92.68 | 93.43 | **91.29**   |

其中，`All Average`显示的是在所有测试集上F-score的平均。

更多详细比较可参见[和现有工具包的比较](https://github.com/lancopku/pkuseg-python/wiki/%E5%92%8C%E7%8E%B0%E6%9C%89%E5%B7%A5%E5%85%B7%E5%8C%85%E7%9A%84%E6%AF%94%E8%BE%83)。

## 使用方式

#### 代码示例

以下代码示例适用于python交互式环境。

代码示例1：使用默认配置进行分词（**如果用户无法确定分词领域，推荐使用默认模型分词**）
```python3
import pkuseg

seg = pkuseg.pkuseg()           # 以默认配置加载模型
text = seg.cut('我爱北京天安门')  # 进行分词
print(text)
```

代码示例2：细领域分词（**如果用户明确分词领域，推荐使用细领域模型分词**）
```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
text = seg.cut('我爱北京天安门')              # 进行分词
print(text)
```

代码示例3：分词同时进行词性标注，各词性标签的详细含义可参考 [tags.txt](https://github.com/lancopku/pkuseg-python/blob/master/tags.txt)
```python3
import pkuseg

seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
text = seg.cut('我爱北京天安门')    # 进行分词和词性标注
print(text)
```


代码示例4：对文件分词
```python3
import pkuseg

# 对input.txt的文件分词输出到output.txt中
# 开20个进程
pkuseg.test('input.txt', 'output.txt', nthread=20)     
```


代码示例5：额外使用用户自定义词典
```python3
import pkuseg

seg = pkuseg.pkuseg(user_dict='my_dict.txt')  # 给定用户词典为当前目录下的"my_dict.txt"
text = seg.cut('我爱北京天安门')                # 进行分词
print(text)
```


代码示例6：使用自训练模型分词（以CTB8模型为例）
```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='./ctb8')  # 假设用户已经下载好了ctb8的模型并放在了'./ctb8'目录下，通过设置model_name加载该模型
text = seg.cut('我爱北京天安门')            # 进行分词
print(text)
```



代码示例7：训练新模型 （模型随机初始化）
```python3
import pkuseg

# 训练文件为'msr_training.utf8'
# 测试文件为'msr_test_gold.utf8'
# 训练好的模型存到'./models'目录下
# 训练模式下会保存最后一轮模型作为最终模型
# 目前仅支持utf-8编码，训练集和测试集要求所有单词以单个或多个空格分开
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')	
```


代码示例8：fine-tune训练（从预加载的模型继续训练）
```python3
import pkuseg

# 训练文件为'train.txt'
# 测试文件为'test.txt'
# 加载'./pretrained'目录下的模型，训练好的模型保存在'./models'，训练10轮
pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
```



#### 参数说明

模型配置
```
pkuseg.pkuseg(model_name = "default", user_dict = "default", postag = False)
	model_name		模型路径。
			        "default"，默认参数，表示使用我们预训练好的混合领域模型(仅对pip下载的用户)。
				"news", 使用新闻领域模型。
				"web", 使用网络领域模型。
				"medicine", 使用医药领域模型。
				"tourism", 使用旅游领域模型。
			        model_path, 从用户指定路径加载模型。
	user_dict		设置用户词典。
				"default", 默认参数，使用我们提供的词典。
				None, 不使用词典。
				dict_path, 在使用默认词典的同时会额外使用用户自定义词典，可以填自己的用户词典的路径，词典格式为一行一个词。
	postag		        是否进行词性分析。
				False, 默认参数，只进行分词，不进行词性标注。
				True, 会在分词的同时进行词性标注。
```

对文件进行分词
```
pkuseg.test(readFile, outputFile, model_name = "default", user_dict = "default", postag = False, nthread = 10)
	readFile		输入文件路径。
	outputFile		输出文件路径。
	model_name		模型路径。同pkuseg.pkuseg
	user_dict		设置用户词典。同pkuseg.pkuseg
	postag			设置是否开启词性分析功能。同pkuseg.pkuseg
	nthread			测试时开的进程数。
```

模型训练
```
pkuseg.train(trainFile, testFile, savedir, train_iter = 20, init_model = None)
	trainFile		训练文件路径。
	testFile		测试文件路径。
	savedir			训练模型的保存路径。
	train_iter		训练轮数。
	init_model		初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'。
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



## 预训练模型

从pip安装的用户在使用细领域分词功能时，只需要设置model_name字段为对应的领域即可，会自动下载对应的细领域模型。

从github下载的用户则需要自己下载对应的预训练模型，并设置model_name字段为预训练模型路径。预训练模型可以在[release](https://github.com/lancopku/pkuseg-python/releases)部分下载。以下是对预训练模型的说明：

- **news**: 在MSRA（新闻语料）上训练的模型。

- **web**: 在微博（网络文本语料）上训练的模型。

- **medicine**: 在医药领域上训练的模型。

- **tourism**: 在旅游领域上训练的模型。

- **mixed**: 混合数据集训练的通用模型。随pip包附带的是此模型。



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
- v0.0.18(2019-02-20)
  - 支持词性标注，增加了医疗、旅游两个细领域模型


## 开源协议
1. 本代码采用MIT许可证。
2. 欢迎对该工具包提出任何宝贵意见和建议，请发邮件至jingjingxu@pku.edu.cn。



## 相关论文

该代码包主要基于以下科研论文，如使用了本工具，请引用以下论文：
* Xu Sun, Houfeng Wang, Wenjie Li. [Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection](http://www.aclweb.org/anthology/P12-1027). ACL. 253–262. 2012 

```
@inproceedings{DBLP:conf/acl/SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea- Volume 1: Long Papers},
pages = {253--262},
year = {2012}}
```



## 常见问题及解答


1. [为什么要发布pkuseg？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#1-为什么要发布pkuseg)
2. [pkuseg使用了哪些技术？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#2-pkuseg使用了哪些技术)
3. [无法使用多进程分词和训练功能，提示RuntimeError和BrokenPipeError。](https://github.com/lancopku/pkuseg-python/wiki/FAQ#3-无法使用多进程分词和训练功能提示runtimeerror和brokenpipeerror)
4. [是如何跟其它工具包在细领域数据上进行比较的？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#4-是如何跟其它工具包在细领域数据上进行比较的)
5. [在黑盒测试集上进行比较的话，效果如何？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#5-在黑盒测试集上进行比较的话效果如何)
6. [如果我不了解待分词语料的所属领域呢？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#6-如果我不了解待分词语料的所属领域呢)
7. [如何看待在一些特定样例上的分词结果？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#7-如何看待在一些特定样例上的分词结果)
8. [关于运行速度问题？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#8-关于运行速度问题)
9. [关于多进程速度问题？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#9-关于多进程速度问题)



## 作者

Ruixuan Luo （罗睿轩）,  Jingjing Xu（许晶晶）, Xuancheng Ren（任宣丞）, Yi Zhang（张艺）, Bingzhen Wei（位冰镇）， Xu Sun （孙栩）



# Pkuseg 

A multi-domain Chinese word segmentation toolkit.

## Highlights

The pkuseg-python toolkit has the following features:

1.	Supporting multi-domain Chinese word segmentation. Pkuseg-python supports multi-domain segmentation, including domains like news, web, medicine, and tourism. Users are free to choose different pre-trained models according to the domain features of the text to be segmented. If not sure the domain of the text, users are recommended to use the default model trained on mixed-domain data.

2.	Higher word segmentation results. Compared with existing word segmentation toolkits, pkuseg-python can achieve higher F1 scores on the same dataset.

3.	Supporting model training. Pkuseg-python  also supports users to train a new segmentation model with their own data.

4.	Supporting POS tagging. We also provide users POS tagging interfaces for further lexical analysis. 



## Installation

- Requirements: python3

1. Install pkuseg-python by using PyPI: (with the default model trained on mixed-doimain data)
	```
	pip3 install pkuseg
	```
   or update to the latest version (**suggested**):
   	```
	pip3 install -U pkuseg
	```
2. Install pkuseg-python by using image source for fast speed:
	```
	pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pkuseg
	```
   or update to the latest version (**suggested**):
	```
	pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pkuseg
	```
   Note: The previous two installing commands only support python3.5, python3.6, python3.7 on linux, mac, and **windows 64 bit**.
3. If the code is downloaded from GitHub, please run the following command to install pkuseg-python:
	```
	python setup.py build_ext -i
	```
	
   Note: the github code does not contain the pre-trained models, users need to download the pre-trained models from [release](https://github.com/lancopku/pkuseg-python/releases), and set parameter 'model_name' as the model path.
   
   
	

## Usage

#### Examples


Example 1:	Segmentation under the default configuration. **If users are not sure the domain of the text to be segmented, the default configuration is recommended.**
```python3
import pkuseg

seg = pkuseg.pkuseg() #load the default model
text = seg.cut('我爱北京天安门')
print(text)
```

Example 2: Domain-specific segmentation. **If users know the text domain, they can select a pre-trained domain model according to the domain features.**

```python3
import pkuseg
seg = pkuseg.pkuseg(model_name='medicine') 
#Automatically download the domain-specific model.
text = seg.cut('我爱北京天安门')
print(text)
```

Example 3：Segmentation and POS tagging. For the detailed meaning of each POS tag, please refer to [tags.txt](https://github.com/lancopku/pkuseg-python/blob/master/tags.txt).
```python3
import pkuseg

seg = pkuseg.pkuseg(postag=True)                           
text = seg.cut('我爱北京天安门')
print(text)
```


Example 4：Segmentation with a text file as input.
```python3
import pkuseg

#Take file 'input.txt' as input. 
#The segmented result is stored in file 'output.txt'.
pkuseg.test('input.txt', 'output.txt', nthread=20)     
```


Example 5: Segmentation with a user-defined dictionary.
```python3
import pkuseg

seg = pkuseg.pkuseg(user_dict='my_dict.txt')
text = seg.cut('我爱北京天安门')
print(text)
```


Example 6: Segmentation with a user-trained model. Take CTB8 as an example.
```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='./ctb8') 
text = seg.cut('我爱北京天安门')
print(text)
```



Example 7: Training a new model (randomly initialized).

```python3
import pkuseg

# Training file: 'msr_training.utf8'.
# Test file: 'msr_test_gold.utf8'.
# Save the trained model to './models'.
# The training and test files are in utf-8 encoding.
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')	
```

Example 8: Fine-tuning. Take a pre-trained model as input.
```python3
import pkuseg

# Training file: 'train.txt'.
# Testing file'test.txt'.
# The path of the pre-trained model: './pretrained'.
# Save the trained model to './models'.
# The training and test files are in utf-8 encoding.
pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
```



#### Parameter Settings

Segmentation for sentences.
```
pkuseg.pkuseg(model_name = "default", user_dict = "default", postag = False)
	model_name		The path of the used model.
			        "default". The default mixed-domain model.
				"news". The model trained on news domain data.
				"web". The model trained on web domain data.
				"medicine". The model trained on medicine domain data.
				"tourism". The model trained on tourism domain data.
			        model_path. Load a model from the user-specified path.
	user_dict		Set up the user dictionary.
				"default". Use the default dictionary.
				None. No dictionary is used.
				dict_path. The path of the user-defined dictionary. Each line only contains one word.
	postag		        POS tagging or not.
				False. The default setting. Segmentation without POS tagging.
				True. Segmentation with POS tagging.
```

Segmentation for documents.

```
pkuseg.test(readFile, outputFile, model_name = "default", user_dict = "default", postag = False, nthread = 10)
	readFile		The path of the input file.
	outputFile		The path of the output file.
	model_name		The path of the used model. Refer to pkuseg.pkuseg.
	user_dict		The path of the user dictionary. Refer to pkuseg.pkuseg.
	postag			POS tagging or not. Refer to pkuseg.pkuseg.
	nthread			The number of threads.
```

 Model training.
```
pkuseg.train(trainFile, testFile, savedir, train_iter = 20, init_model = None)
	trainFile		The path of the training file.
	testFile		The path of the test file.
	savedir			The saved path of the trained model.
	train_iter		The maximum number of training epochs.
	init_model		By default, None means random initialization. Users can also load a pre-trained model as initialization, like init_model='./models/'.
```








