# SwinJSCC Reimplementation
An unofficial implementation of SwinJSCC. It builds on the official repository, restructures the code, corrects several issues, and adds more explanatory annotations.

This work mainly refines the **data loading mechanism, the hyperparameter transmission method, and the learning rate update process** within the official implementation. The system enables dataset selection via hyperparameter specification and this eliminates the necessity of manual list modification. The command line facilitates the transmission of all hyperparameters and supplants the manual assignment typically associated with the static `config` class. Furthermore, the architecture incorporates a learning rate decay module and replaces the constant learning rate strategy. Given that the RA and SA modules in SwinJSCC induce performance degradation and compromise training stability, the revised framework excludes these components. Consequently, the design retains only the core encoder, channel, and decoder modules. Additionally, the integration of `tensorboard` logging facilitates the visualization of the training trajectory.

Official Repository：https://github.com/semcomm/SwinJSCC

Original Paper：https://arxiv.org/abs/2308.09361

SwinJSCC的非官方实现。基于官方的代码仓库，进行了代码重构，修改了部分bug，增添了更多的注释说明内容。

## 重构

对`main.py`部分进行了大幅修改，其他部分进行了代码重构。主要包含以下内容：

1. 将所有训练参数以命令行的形式传递，修改了原代码中通过`config`类手动指定的行为，使网络的训练与参数修改更容易，超参数的选择更清晰
2. 在原始代码中，必须用列表传递数据集。该版本将其更改为了用命令行参数，支持用户自选数据集
3. 修改了优化器。添加了学习率衰减模块，`build_scheduler`，用户可设置`warmup_epochs `和`hold_epochs`，学习率衰减使用的是余弦衰减
4. 添加了`tensorboard`日志，支持训练过程的可视化

对其他模块的修改如下：

1. 删除去原始论文中的`SA`和`RA`部分。这两个模块的设计相当不合理，大量的MLP融合操作会导致训练不稳定，易发生梯度中断，而且无法提供图像重构的增益，反而会导致PSNR下降。如想添加信道和速率适配模块，可参考https://github.com/dccc2025/SwinJSCC-f中的实现

2. 由于仅删除了`SA`和`RA`，对模型的整体框架没有改变，且在模型加载阶段，已设置`strict=False`，因此可直接加载官方的预训练权重

3. 添加了注释以解释代码的含义

4. 删除了`config`类，所有参数通过命令行的`args`传递，包括模型的各类参数

5. 在`datasets.py`文件下，修改了官方代码的一个bug。官方代码会检查输入图像是否是128的倍数，128来自于图像每次下采样的尺寸缩放、窗口大小的选择多，以及`patch`维度的选择。我们将其修改为动态的，代码能够根据相关参数自动计算出最小图像尺寸，进而执行更精准的图像尺度检查，具体修改代码如下：

   ~~~python
   image = Image.open(image_ori).convert('RGB')
           self.im_height, self.im_width = image.size
           if self.im_height % self.mini_size != 0 or self.im_width % self.mini_size != 0:
               self.im_height = self.im_height - self.im_height % self.mini_size
               self.im_width = self.im_width - self.im_width % self.mini_size
   ~~~

   

## 运行

官方提到，PyTorch在大于`1.12`时性能会比论文有一定程度的下降，在复现过程中我们也发现了这一问题，我们没有解决该问题。

在`train.sh`中，我们给出了训练的命令示例，可直接执行该命令：

~~~
python main.py --pass_channel True --training True --distortion_metric MSE --channel_type awgn --C 96 --multiple_snr 1,3,5,7
~~~

如果想自定义模型的参数，**请注意**模型的`patch`必须是2，这是由解码器的`update_resolution`函数决定的，该函数需要更新重构过程中图像的维度，但官方代码默认`patch`是2，如果想自定义`patch`，需要修改该函数：

~~~python
def update_resolution(self, H, W):
        # 这个函数约束了整个网络的patch必须是2的倍数，否则这里恢复的H和W就会出问题
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))
~~~

在`test.sh`中，我们给出了测试的命令示例，注意需要有预训练好的模型，这里默认预训练模型存放在当前目录下：

~~~
python main.py --distortion_metric MSE \
--channel_type awgn --C 96 --multiple_snr 1,3,5,7 --training False \
--model_path SwinJSCC_Channel_EP1500_c96.model --pass_channel True \
--test_data_dir /home/serika/Kodak/
~~~

对于论文中的传统图像压缩算法，可通过CompressAI的相关API实现：https://github.com/InterDigitalInc/CompressAI



## 相关链接

官方代码实现：https://github.com/semcomm/SwinJSCC

原始论文：https://arxiv.org/abs/2308.09361

`DIV2K`数据集: https://data.vision.ee.ethz.ch/cvl/DIV2K/

`Kodak`数据集: http://r0k.us/graphics/kodak/

`CLIC`数据集: [http://compression.cc](http://compression.cc/)
