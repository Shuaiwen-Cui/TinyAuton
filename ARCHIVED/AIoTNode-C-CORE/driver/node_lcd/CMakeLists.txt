set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    esp_driver_gpio
    esp_driver_spi
    node_spi
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})