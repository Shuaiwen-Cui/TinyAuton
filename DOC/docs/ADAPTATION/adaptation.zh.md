# 适配

!!! info "关于适配层"
    适配层是一个介于MCU平台与TinyAuton之间的缓冲层，其核心作用是将平台特有的部分与TinyAuton中关于计算与智能的部分进行解耦，从而方便TinyAuton的跨平台移植与应用。

!!! warning     
    目前开发以ESP32为基础，向STM32等平台的迁移需要对适配层进行一定的修改。

## ESP32官方库的支持

<div class="grid cards" markdown>

-   :simple-espressif:{ .lg .middle } __ESP-DSP__

    ---

    [:octicons-arrow-right-24: <a href="https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/index.html" target="_blank"> 在线文档 </a>](#)

-   :simple-espressif:{ .lg .middle } __ESP-DL__

    ---

    [:octicons-arrow-right-24: <a href="https://docs.espressif.com/projects/esp-dl/zh_CN/latest/getting_started/readme.html" target="_blank"> Online Doc </a>](#)

</div>