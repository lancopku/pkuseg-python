
# 多进程分词

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
