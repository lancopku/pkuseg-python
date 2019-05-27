# 代码示例

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
