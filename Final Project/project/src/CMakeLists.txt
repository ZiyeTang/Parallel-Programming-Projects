aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)

FILE(GLOB cuSources ./layer/custom/*.cu)
cuda_add_library(GpuConv ${cuSources})
#cuda_add_library(GpuConv ./layer/custom/new-forward.cu ./layer/custom/gpu-utils.cu)
# set_target_properties(GpuConv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library (MiniDNNLib ${DIR_LIB_SRCS} ./layer/custom/cpu-new-forward.cc)
target_link_libraries(MiniDNNLib GpuConv)


