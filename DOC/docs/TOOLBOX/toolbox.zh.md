# 工具箱

!!! info "tiny_toolbox"
    工具箱tiny_toolbox定位是用于 **平台适配与优化** 并提供 **各种实用工具** 的库，服务于边缘计算与应用开发。**注意，之所以将适配和工具放在一个库里面，是因为很多工具底层利用的是平台提供的功能，所以将平台适配和各类工具放在同一个库里面，便于使用和维护。**

!!! warning     
    目前开发以ESP32为基础，向STM32等平台的迁移需要对适配层进行一定的修改。

## 组件依赖

```c
set(src_dirs
    .
    time
)

set(include_dirs
    .
    time
)

set(requires
    esp_timer
    esp_rtc
    espressif__esp-dsp
    espressif__esp_jpeg
    espressif__esp-dl
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})


```

## 架构与功能目录

```txt
    tiny_toolbox
    ├── CMakeLists.txt
    ├── tiny_toolbox.h // serves as a directory, integrating all submodules
    ├── time
    │   ├── tiny_time.h // submodule for time management - header file
    │   ├── tiny_time.c // submodule for time management - source file
    │   └── ...
    └── ...
```

### [时间](./TIME/notes.zh.md)

- 获取运行时间： `tiny_get_running_time()`
- SNTP对时： `sync_time_with_timezone("CST-8")`
- 获取世界时间： `tiny_get_current_time(1)`

待开发:

- 无线传感器网络本地对时-微秒级别

## 代码

!!! tip
    tiny_toolbox.h 只是作为一个目录，集成了所有的子模块，具体的功能在各个子模块中实现。tiny_toolbox.c 只是形式上的源文件，没有具体的功能。

### CMakeLists.txt

```cmake
set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    esp_timer
    espressif__esp-dsp
    espressif__esp_jpeg
    espressif__esp-dl
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})
```

### tiny_toolbox.h

```c
/**
 * @file tiny_toolbox.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the tiny_toolbox middleware.
 * @version 1.0
 * @date 2025-03-26
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINY_TOOLBOX__
#define __TINY_TOOLBOX__

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */
// system
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// customized drivers
#include "esp_rtc.h"

/* SUBMODULES */
#include "tiny_time.h" // Time

#ifdef __cplusplus
}
#endif

#endif /* __TINY_TOOLBOX__ */
```
