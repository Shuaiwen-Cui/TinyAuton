set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    driver
    esp_driver_gpio
    esp_timer
    node_led
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})