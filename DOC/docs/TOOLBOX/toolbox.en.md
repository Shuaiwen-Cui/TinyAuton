# TOOLBOX

!!! note "tiny_toolbox"
    tiny_toolbox is a library designed for **platform adaptation and optimization**, providing **various practical tools** to serve edge computing and application development. **Note that the adaptation and tools are included in the same library because many tools utilize the functions provided by the platform at a lower level. Therefore, placing platform adaptation and various tools together facilitates usage and maintenance.**

!!! warning
    Currently, development is based on ESP32, and migration to platforms like STM32 requires some modifications to the adaptation layer.

## COMPONENT DEPENDENCIES

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

## ARCHITECTURE AND FUNCTION DIRECTORY

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

### [TIME](./TIME/notes.en.md)

- Get Running Time: `tiny_get_running_time()`
- SNTP Time Synchronization: `sync_time_with_timezone("CST-8")`
- Get World Time: `tiny_get_current_time(1)`
  
TODO:

- Local Time Synchronization for Wireless Sensor Networks - Microsecond Level

## CODE

!!! tip
    tiny_toolbox.h serves merely as a directory, integrating all submodules. The specific functionalities are implemented in each submodule. tiny_toolbox.c is just a formal source file without specific functionality.

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
