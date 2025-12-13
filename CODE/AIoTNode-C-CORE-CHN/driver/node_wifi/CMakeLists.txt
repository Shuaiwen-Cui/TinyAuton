set(src_dirs
    .
)

set(include_dirs
    include
)

set(requires
    node_led
    node_lcd
    esp_wifi
    wpa_supplicant
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})