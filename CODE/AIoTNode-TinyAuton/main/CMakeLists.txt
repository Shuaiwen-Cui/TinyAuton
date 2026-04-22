# Define source directories
set(src_dirs
    .
)

# Define include directories
set(include_dirs
    .
)

# Register the component
idf_component_register(
    SRC_DIRS ${src_dirs}
    INCLUDE_DIRS ${include_dirs}
    REQUIRES
        freertos
        log
        esp_system
        nvs_flash
        spi_flash
        esp_psram
        esp_wifi
        node_espnow
        node_led
        node_exit
        node_spi
        node_timer
        node_rtc
        node_sdcard
)

# Add compilation options
# component_compile_options(-ffast-math -O3 -Wno-error=format -Wno-format)
