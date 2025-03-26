# ARCHITECTURE

## LAYERED ARCHITECTURE

```txt
+------------------------------+
| AI                           | <-- AI/ML Functions for Edge Devices based on Low Level Functions
+------------------------------+
| DSP                          | <-- Digital Signal Processing Functions
+------------------------------+
| Math Operations              | <-- Commonly Used Math Functions for Various Applications
+------------------------------+
| Adaptation Layer             | <-- To Replace Functions in Standard C with Platform Optimized/Specific Functions
+------------------------------+
```

## CODE ORGANIZATION

```txt

+------------------------------+
| APPLICATION                  |
+------------------------------+
| MIDDLEWARE                   |
|   - TinyAdapter              | <-- To shield HW differences and provide a unified API
|   - TinyMath                 | <-- Common Math Functions
|   - TinyDSP                  | <-- DSP Functions
|   - TinyAI                   | <-- AI Functions
|   - TinyEvaluator            | <-- To Evaluate Performance
+------------------------------+
| DRIVERS                      |
+------------------------------+
| HARDWARE                     |
+------------------------------+

```