# 使用说明

!!! info "使用说明"
    该文档是对 `tiny_math` 模块的使用说明。

## 整体引入TinyMath

!!! info
    适用于C项目，或者结构较为简单的C++项目。

```c
#include "tiny_math.h"
```

## 分模块引入TinyMath

!!! info
    适用于需要精确控制引入模块的项目，或者复杂的C++项目。

```c
#include "tiny_vec.h" // 引入向量模块
#include "tiny_mat.h" // 引入矩阵模块
```

```cpp
#include "tiny_matrix.hpp" // 引入高级矩阵模块
```

!!! tip "注意"

    - `tiny_vec.h` 和 `tiny_mat.h` 是 C 语言版本的头文件，适用于 C 语言编程。

    - `tiny_matrix.hpp` 是 C++ 语言版本的头文件，适用于 C++ 语言编程。

    简单来说，C语言项目只能用 `tiny_vec.h` 和 `tiny_mat.h`，而 C++ 项目可以使用 `tiny_vec.h`、`tiny_mat.h` 和 `tiny_matrix.hpp`。

!!! tip
    具体的使用方法请参考测试代码。