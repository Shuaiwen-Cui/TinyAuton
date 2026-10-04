# LOG {#log}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

> 2025-04-10

- 获取运行时间： `tiny_get_running_time()`
- SNTP对时： `sync_time_with_timezone("CST-8")`
- 获取世界时间： `tiny_get_current_datetime(1)`

待开发:

- 无线传感器网络本地对时-微秒级别
