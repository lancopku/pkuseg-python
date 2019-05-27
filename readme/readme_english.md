
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


## Publication

The toolkit is mainly based on the following publication. If you use the toolkit, please cite the paper:
* Xu Sun, Houfeng Wang, Wenjie Li. [Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection](http://www.aclweb.org/anthology/P12-1027). Proceedings of ACL. 253–262. 2012 

```
@inproceedings{SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {Proceedings of ACL},
pages = {253--262},
year = {2012}}
```



## Authors

Ruixuan Luo, Jingjing Xu, Xuancheng Ren, Yi Zhang, Bingzhen Wei, Xu Sun  

[Language Computing and Machine Learning Group](http://lanco.pku.edu.cn/), Peking University


