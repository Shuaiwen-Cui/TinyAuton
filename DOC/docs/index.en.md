# TINYAUTON

![cover](cover.jpg)

## ABOUT THIS PROJECT

This project dedicates to the development of a library for tiny agent related computing running on MCU devices to serve the multi-agent system，covering mathematical operations, digital signal processing, and TinyML. 

!!! info "About the Name"
    The name "TinyAuton" is a combination of "Tiny" and "Auton". "Tiny" means the agent is designed to run on MCU devices, and "Auton" is short for "Autonomous Agent".

## TARGET HARDWARE

- General MCU Devices (Standard C)

## PLATFORM OPTIMIZATION

- ARM Cortex Series
- ESP32 Series

## SCOPE

- Basic Math Operations
- Digital Signal Processing
- TinyML / Edge AI

## ARCHITECTURE

```txt
+------------------------------+
| TinyML                       | <-- AI/ML Functions for Edge Devices based on Low Level Functions
+------------------------------+
| DSP                          | <-- Digital Signal Processing Functions
+------------------------------+
| Math Operations              | <-- Commonly Used Math Functions for Various Applications
+------------------------------+
| Adaptation Layer (AL)        | <-- To Replace Functions in Standard C with Platform Optimized/Specific Functions
+------------------------------+
```

## DEVELOPMENT HOST MCU HARDWARE

- DNESP32S3M from Alientek (ESP32-S3)
- FK743M2-IIT6 from FANKE (STM32H743)