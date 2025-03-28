# ADAPTATION

!!! note "ADAPTATION"
    The adaptation layer serves as a buffer layer between the MCU platform and TinyAuton. Its core function is to decouple the platform-specific components from the computational and intelligent components in TinyAuton, facilitating cross-platform migration and application of TinyAuton. 

!!! warning
    Currently, development is based on ESP32, and migration to platforms like STM32 will require some modifications to the adaptation layer.