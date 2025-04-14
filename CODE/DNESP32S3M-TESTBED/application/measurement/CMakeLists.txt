set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    driver
    tim
    esp_rtc
    spi_sdcard
    mpu6050
    exit
    lcd
    led
    wifi
    esp32_mqtt
    TinyToolbox
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})