# ESP-DSP

<div class="grid cards" markdown>

-   :simple-espressif:{ .lg .middle } __ESP-DSP__

    ---

    An Espressif DSP Library (esp-dsp) it’s library of functions, modules and components that provides possibility to use Espressif’s CPUs as DSPs in efficient way.

    [:octicons-arrow-right-24: <a href="https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/index.html" target="_blank"> Online Doc </a>](#)

</div>


## Function Naming

Naming conventions for the Library functions are similar for all covered domains. You can distinguish signal processing functions by the dsps prefix, while image and video processing functions have dspi prefix, and functions that are specific for operations on small matrices have dspm prefix in their names. Function names in Library have the following general format:

```txt
dsp<data-domain>_<name>_<datatype1><datatype_ext>_<datatype2><datatype_ext>[_<descriptor>]<_impl>(<parameters>);
```

Where:

-   `<data-domain>` is the domain of the function, e.g. `s` for signal processing, `i` for image processing, `v` for video processing, and `m` for small matrix operations.
  
-   `<name>` is the name of the function.
   
-   `<datatype1>` is the type of the first input parameter.
   
-   `<datatype_ext>` is the type of the first input parameter extended with a suffix that indicates the type of the data, e.g. `f` for float, `i` for integer, `c` for complex, etc.
   
-   `<datatype2>` is the type of the second input parameter.
   
-   `<descriptor>` is an optional descriptor that provides additional information about the function.
   
-   `<impl>` is an optional implementation descriptor that provides additional information about the implementation of the function.
   
-   `<parameters>` are the parameters of the function.


## Data Domain

The data-domain is a single character that expresses the subset of functionality to which a given function belongs. The Library designed to supports the following data-domains:

- s - for signals (expected data type is a 1D signal)

- i - for images and video (expected data type is a 2D image)

- m - for matrices (expected data type is a matrix)

- r - for realistic rendering functionality and 3D data processing (expected data type depends on supported rendering techniques)

- q - for signals of fixed length

For example, function names that begin with dspi signify that respective functions are used for image or video processing.

## Name

The name is an abbreviation for the core operation that the function really does, for example Add, Sqrt, followed in some cases by a function-specific modifier: = [_modifier]

This modifier, if present, denotes a slight modification or variation of the given function.

## Data Types

The library supports two main data types – int16 for fixed point arithmetic and float for floating point arithmetic. The datatype described as:

### Data type suffices:

- s - signed

- u - unsigned

- f - float

### Data type extensions:

- c - complex

### Data type Bits resolution:

- 16

- 32

For example: dsps_mac_sc16 defines that mac operation with 1d array will be made with 16 bit signed complex data.

## Implementation Type

Each function could be implemented different for different platform and could use different style and resources. That’s why every implemented function will have name extension <_impl> that will define which kind of implementation it is. User can use universal function without extension.

### Implementation extensions:

By default all functions could be used without extensions. The option that select optimized/ansi can be chosen in menuconfig.

Inside library the extensions means:

- _ansi - a universal function where body of function implemented on ANSI C. This implementation not includes any hardware optimization

- _ae32 - written on ESP32 assembler and optimized for ESP32

- _aes3 - written on ESP32S3 assembler and optimized for ESP32S3

- _arp4 - written on ESP32P4 assembler and optimized for ESP32P4

- _platform - header file with definitions of available CPUs instructions for different functions

- others- depends on amount of supported CPUs. This list will be extended in future