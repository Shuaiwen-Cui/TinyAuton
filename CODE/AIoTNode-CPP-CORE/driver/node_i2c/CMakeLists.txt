set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    driver
    esp_driver_i2c
    log
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})