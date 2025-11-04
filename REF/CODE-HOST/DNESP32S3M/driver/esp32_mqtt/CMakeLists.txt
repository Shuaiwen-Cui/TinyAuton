set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    # led
    # lcd
    # esp_wifi
    # wpa_supplicant
    # wifi
    mqtt
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})