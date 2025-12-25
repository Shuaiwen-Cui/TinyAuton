set(src_dirs
    .
    core
    operator
    loss
    model
    train
    test
)

set(include_dirs
    .
    include
    core
    operator
    loss
    model
    train
    test
)

set(requires
    tiny_math
    tiny_dsp
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

