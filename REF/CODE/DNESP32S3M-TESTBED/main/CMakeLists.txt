# Define source directories
set(src_dirs
    .
)

# Define include directories
set(include_dirs
    .
)

# Define required components
set(requires
)

# Register the component
idf_component_register(
    SRC_DIRS ${src_dirs}
    INCLUDE_DIRS ${include_dirs}
    REQUIRES ${requires}
)

# Add compilation options
# component_compile_options(-ffast-math -O3 -Wno-error=format -Wno-format)