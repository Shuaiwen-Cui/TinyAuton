# 前置条件

## 硬件与软件要求

ESP32S3 开发板, 推荐参考以下项目:

<div class="grid cards" markdown>

-   :simple-github:{ .lg .middle } __NexNode__

    ---

    [:octicons-arrow-right-24: <a href="https://github.com/Shuaiwen-Cui/NexNode.git" target="_blank"> 代码 </a>](#)

    [:octicons-arrow-right-24: <a href="http://www.cuishuaiwen.com:9100/" target="_blank"> 文档 </a>](#)


</div>

我们以该项目中的代码为基础进行进一步开发。

## 依赖组件

为了提升我们框架的计算效率，我们首先引入ESP-DSP库和ESP-DL库，它们分别提供了数字信号处理和深度学习相关的高效实现。

!!! TIP
    注意以上两个库似乎是由不同团队开发，因此他们的很多功能有重叠。

```txt
- espressif__esp-dsp
- espressif__esp-dl
   - espressif__dl_fft
   - espressif__esp_new_jpeg
```

我们可以在ESP-REGISTRY中找到和下载这些组件到项目中。在本项目中我将下载的组件及其依赖组件移动到了`middleware`文件夹下，并移除了配置文件，从而避免版本锁定和网络依赖。