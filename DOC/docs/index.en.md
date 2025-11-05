# TINYAUTON: Microcontroller-oriented Distributed Intelligence Enabling Framework

![cover](cover.jpg)

## ABOUT THIS PROJECT

This project dedicates to the development of a library for tiny agent related computing running on MCU devices to serve the multi-agent system，covering mathematical operations, digital signal processing, and TinyML. 

!!! info "About the Name"
    The name "TinyAuton" is a combination of "Tiny" and "Auton". "Tiny" means the agent is designed to run on MCU devices, and "Auton" is short for "Autonomous Agent".

## TARGET HARDWARE

- MCU devices (currently targeting ESP32 as the main platform)

## SCOPE

- Platform adaptation and various tools (time, communication, etc.)
- Basic Math Operations
- Digital Signal Processing
- TinyML / Edge AI


## HOST DEVKITS

!!! TIP 
    The following hardwares are for demonstration purposes only. This project is not limited to these and can be ported to other types of hardwares.

- DNESP32S3M from Alientek (ESP32-S3)

![DNESP32S3M](DNESP32S3M.png){width=800px}

![DNESP32S3M-BACK](DNESP32S3M-BACK.png){width=800px}

<div class="grid cards" markdown>

-   :simple-github:{ .lg .middle } __NexNode__

    ---

    [:octicons-arrow-right-24: <a href="https://github.com/Shuaiwen-Cui/NexNode.git" target="_blank"> Repo </a>](#)

    [:octicons-arrow-right-24: <a href="http://www.cuishuaiwen.com:9100/" target="_blank"> Online Doc </a>](#)


</div>

## PROJECT ARCHITECTURE

```txt
+------------------------------+
| APPLICATION                  |
+------------------------------+
|   - TinyAI                   | <-- AI Functions
|   - TinyDSP                  | <-- DSP Functions
|   - TinyMath                 | <-- Common Math Functions
|   - TinyToolbox              | <-- Platform-specific Low-level Optimization + Various Utilities
| MIDDLEWARE                   |
+------------------------------+
| DRIVERS                      |
+------------------------------+
| HARDWARE                     |
+------------------------------+
```