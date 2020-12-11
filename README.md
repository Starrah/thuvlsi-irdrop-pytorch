基于Pytorch的静态IR-Drop分析
===============================
清华大学计算机系VLSI设计导论大作业
## 说明
本代码库中只带了一组比较小的数据：ibmpg1.spice及其solution。同时携带了我利用该数据跑出来的一组结果。如果需要更多的数据，可以从https://web.ece.ucsb.edu/~lip/PGBenchmarks/ibmpgbench.html下载。  
实验输入：普通的Netlist网表文件(.spice)，groundtruth文件(.solution)。如果没有groundtruth，可以把main.py第12行`n.load_groundtruth`注释掉，这样就不会利用groundtruth计算RMSE了。  
代码运行的产物：.out文件，每一行都是`节点名 电压`的格式。  
## 运行
依赖：
- Python 3.8（较低版本应该也可）
- Pytorch
- Matplotlib
- CUDA（可选，如果不想用GPU运行，可以把utils.py第8行的`global_device`改为`torch.device("cpu")`，但是可能要解很久。）
运行：
```shell script
python3 main.py
```
配置数据：
在main.py的第34行左右指定路径。main.py的16-20行选定要运行什么solver。
另提供了控制台log可供参考。