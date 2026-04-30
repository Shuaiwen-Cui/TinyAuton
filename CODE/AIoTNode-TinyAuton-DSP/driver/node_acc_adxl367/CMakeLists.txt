# I2C: only driver/node_i2c (no direct esp_driver_i2c use in sources here).
idf_component_register(
    SRCS
        "node_acc_adxl367.c"
        "node_acc_adxl367_test.c"
    INCLUDE_DIRS
        "include"
    REQUIRES
        node_i2c
        freertos
        log
        esp_driver_gpio
)
