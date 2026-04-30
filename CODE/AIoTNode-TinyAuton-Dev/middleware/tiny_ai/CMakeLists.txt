# tiny_ai component CMakeLists.txt

# Source directories — one entry per subdirectory containing .c / .cpp files
set(src_dirs
    .
    core
    layers
    models
    quant
    train
    example
)

# Include directories — exposes headers to this component and its dependents
set(include_dirs
    .
    include
    core
    layers
    models
    quant
    train
    example/data
)

# Required IDF / custom components
set(requires
    tiny_dsp
)

# Register the component
idf_component_register(
    SRC_DIRS    ${src_dirs}
    INCLUDE_DIRS ${include_dirs}
    REQUIRES    ${requires}
)
