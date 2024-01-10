# FwSSNet

# 概述

本文从特征信息的角度探讨了FwSS在BN中的工作机制，提出FwSS通过提取独立特征信息来改善网络的性能。
在此基础上，提出了新的网络层结构bn+FwSS+weights+f(·)，搭建了新神经网络模型FwSSNet
# 环境搭建


** `pip`:**

```bash
pip install -r requirements.txt
```



# 运行说明

##训练FCNNs
修改FCNNs/config中对应数据集配置文件为config.py
```python3
python train.py
```
脚本运行
```bash
bash FCNNs_批量脚本运行.sh
```
##训练CNNs
###vgg
修改CNNs/vgg/config_vggnet.py配置文件
```python3
python train_vggnet.py
```
脚本运行
```bash
bash vgg_批量脚本运行.sh
```
###训练ResNets
修改CNNs/ResNets/config.py配置文件
```python3
python train.py
```
脚本运行
```bash
bash resnets__批量脚本运行.sh
```