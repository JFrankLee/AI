

# 百度网盘AI大赛-文档检测优化赛第8名方案

## 一、简介

生活中人们使用手机进行文档扫描逐渐成为一件普遍的事情，为了提高人们的使用体验，我们期望通过算法技术去除杂乱的拍摄背景并精准框取文档边缘，选手需要通过深度学习技术训练模型，对给定的真实场景下采集得到的带有拍摄背景的文件图片进行边缘智能识别，并最终输出处理后的扫描结果图片。

这个任务可以看为回归问题，也可以堪为分割问题。鉴于现阶段已有众多成熟的分割模型，我们选用了后者。主要网络的构造使用了飞桨PaddlePaddle开发的端到端图像分割开发套件[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，该套件集成了高精度和轻量级等不同方向的大量高质量分割模型。

## 二、模型选择

<img width="401" alt="微信图片_20220524163424" src="https://user-images.githubusercontent.com/60511092/170070265-47328465-6cc3-4dba-9cf6-a32b972e8ec2.png">


PaddleSeg所提供的模型性能如下，我们选取了mIoU最高的SegFormer_B5；mIoU仅次一点，但FLOPs较小的SegFormer_B3以及未在图中所示的轻量级模型PP-LiteSeg进行模型性能测试

我们将比赛所提供的训练集中随机20%作为验证集，这样便于训练时随时监督，进行优化调试。选定最优的模型后再喂入全部训练集得到最终提交的模型。如果上来就把所有训练集都喂入，只依赖A榜测试集来评判模型训练性能太慢了，而且一天最多只能提交5次。

（使用PaddleSeg训练时有个小坑，该套件读取label时按像素值分类，做该任务分割只有两类，即文档与背景，需把png中文档的像素值设为1，而不是通常的255.）

经过多轮训练与测试后，segformer_B3模型最优，经过16000iter后mIoU可以在自己的val上达到0.98，提交A榜的miou是0.9712，模型大小有180M。PP-LiteSeg调整学习率继续训练后可以达到逼近segformer_B3的水平。但提交A榜的miou只有0.96多，该轻量级模型只有58M，有可能是限于参数量泛化性能不高，故最终选择了segformer_B3网络。该网络结构如下图所示
![](https://ai-studio-static-online.cdn.bcebos.com/ca98cea312054c1a9a816387ee02c9e9d381ec5289514f1cb5e3926f867c0146)




## 三、最终结果
在测试时我们选取了80%的数据作为训练集，选定模型之后，我们又将全部数据集喂入模型开始训练，但这次的A榜结果却不如人意，我们对10000iter 12000iter 14000iter 16000iter 18000iter都提交测试，mIoU都没有超过0.97。猜测可能是训练集和A榜测试集数据分布还是有较大的差异，该训练模型过拟合了。最终模型效果图如下所示：

![](https://ai-studio-static-online.cdn.bcebos.com/7bcec947d5fa442fa7772752769b530dfb1491f706b548fbad1102b06ec65cff)

![](https://ai-studio-static-online.cdn.bcebos.com/8c26ea0dc85a4fd5b7fb1dae327a87f3eee49084dd934b4f9b5cc6b87cc6e522)

B榜提交结果为：

![](https://ai-studio-static-online.cdn.bcebos.com/19ad59df865a4c3ab30a58097dbff39dcd0226981e4e404a9f301f704c01c088)




## 四、代码复现

运行predict.py 第一个参数为输入图像路径 第二个参数为输出图像路径

```python
!python predict.py input output
```



