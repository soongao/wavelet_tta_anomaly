# 故事线：从 Reference Gap 到图像条件化正常参照

## 核心命题

CLIP-ZSAD 的关键缺口落在当前测试图像的正常外观参照上。正常/异常文本原型能够形成跨图共享的语义判别轴，但局部异常定位还需要回答另一个问题：在这张图的材质、纹理和结构条件下，什么样的局部响应应当被视为正常。

本文围绕这一 reference gap 构建 image-conditioned normal reference。方法从单张测试图像的 CLIP patch 特征场中选择可靠证据，估计当前图像的正常外观参照，再用该参照重新解释局部语义响应。频域响应在这里刻画 patch 层面的局部结构变化，并进入证据选择过程，使参照估计不只依赖跨图共享的语义响应。

## 需要修正的叙事点

写作时需要承认现有文本原型的可学习属性。在 CLIP-ZSAD 中，文本原型往往已经通过可学习上下文或异常检测目标得到优化。这个事实使 reference gap 更清楚：prompt learning 优化的是正常/异常语义方向的稳定性；当前测试图像的材质、纹理和结构基准仍需要额外估计。

因此，故事线应聚焦于语义参照和外观参照的分工：

正常/异常文本原型回答“局部区域更接近正常语义还是异常语义”。

当前图像的正常参照回答“在这张图的外观上下文中，哪些局部变化仍属于正常范围”。

异常定位需要两者同时成立。缺少第二个参照时，粗糙材质、重复纹理、结构边界和真实缺陷容易落在相近的局部响应模式中。

## 新发现的放置方式

prompt 中加入当前图像的材质、表面纹理或结构描述后，部分样例的定位图会改善：正常纹理和结构边界上的响应减弱，缺陷区域的响应更集中。这个现象说明，外观上下文正在参与局部异常证据的解释。

这个现象的主要价值在于揭示 reference gap。外观描述在文本侧临时补入了当前图像的正常外观信息；本文的方法把这部分信息转到图像侧，从测试图自身估计 image-conditioned normal reference。

## 频域的自然位置

频域的合理位置在 reference gap 之后。叙事顺序是：

1. CLIP-ZSAD 依赖正常/异常语义原型形成跨图共享的语义参照。
2. 工业异常定位依赖局部外观上下文；相似的局部响应在不同材质和结构中含义不同。
3. prompt 加入材质、表面和结构描述后能改变定位结果，说明当前图像的外观参照会影响局部证据解释。
4. 由于测试时没有目标类正常样本，参照需要从单张测试图像内部估计。
5. 参照估计的关键是选择哪些 patch 能代表当前图像的正常外观。
6. patch 特征中的局部结构变化提供了证据选择所需的判据；频域响应用来刻画这种局部变化。

这样，频域响应从 reference gap 自然进入证据选择机制：它帮助区分稳定的正常纹理、结构边界和可能污染正常参照的局部突变。

## 一句话版本

CLIP-ZSAD 的固定或可学习文本原型能够提供正常/异常语义轴，但不能给出当前图像的正常外观基准；当 prompt 中显式加入材质和结构信息后定位改善，说明外观上下文本身影响异常证据的解释。本文从测试图的 patch 特征中估计 image-conditioned normal reference，并用频域响应刻画局部结构变化，使可靠 patch 证据能够支撑当前图像的正常参照。

## 摘要级版本

CLIP-based zero-shot anomaly localization commonly compares visual patches with normal and abnormal text prototypes. These prototypes define a transferable semantic axis, yet local anomaly evidence is also governed by the normal appearance of the current test image. We observe that localization maps can change when image-specific material, surface, or structural descriptions are included in the prompt, indicating that appearance context affects how local responses should be interpreted. Motivated by this reference gap, this study estimates an image-conditioned normal reference from the test image itself. The method selects reliable patch evidence from the CLIP feature field and uses frequency responses to characterize local structural variation during evidence selection. The resulting reference allows local responses to be evaluated relative to the normal appearance of the current image rather than only against shared semantic prototypes.
