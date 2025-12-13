# tiny_ai component CMakeLists.txt

# Define source directories
set(src_dirs
    .
    mlp
    cnn
)

# Define include directories
set(include_dirs
    .
    mlp
    cnn
)

# Define required components
set(requires
    tiny_math
    tiny_dsp
)

# Register the component
idf_component_register(
    SRC_DIRS ${src_dirs}
    INCLUDE_DIRS ${include_dirs}
    REQUIRES ${requires}
)

