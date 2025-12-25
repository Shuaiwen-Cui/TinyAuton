set(src_dirs
    .
    time
)

set(include_dirs
    .
    time
)

set(requires
    esp_timer
    node_rtc
    espressif__esp-dsp
    espressif__esp_new_jpeg
    espressif__dl_fft
    espressif__esp-dl
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})
