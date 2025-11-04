# PREREQUISITES

## HARDWARE AND SOFTWARE REQUIREMENTS

ESP32 development board, please refer to the following projects for details:

<div class="grid cards" markdown>

-   :simple-github:{ .lg .middle } __NexNode__

    ---

    [:octicons-arrow-right-24: <a href="https://github.com/Shuaiwen-Cui/NexNode.git" target="_blank"> Repo </a>](#)

    [:octicons-arrow-right-24: <a href="http://www.cuishuaiwen.com:9100/" target="_blank"> Online Doc </a>](#)


</div>

We will build upon the code in this project for further development.

## DEPENDENCY COMPONENTS

To enhance the computational efficiency of our framework, we first introduce the ESP-DSP library and ESP-DL library, which provide efficient implementations for digital signal processing and deep learning respectively.

!!! TIP
    Note that these two libraries seem to be developed by different teams, so many of their functions overlap.

```txt
- espressif__esp-dsp
- espressif__esp-dl
   - espressif__dl_fft
   - espressif__esp_new_jpeg
```

We can find and download these components from the ESP-REGISTRY into our project. In this project, I moved the downloaded components and their dependencies into the `middleware` folder and removed the configuration files to avoid version locking and network dependencies.