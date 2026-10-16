# Standalone build of the official collector; ROS viewer nodes are not needed.
cmake_minimum_required(VERSION 3.18)
project(yopo_dataset LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(yaml-cpp REQUIRED)

add_library(yopo_maps STATIC src/maps.cpp src/perlinnoise.cpp)
target_include_directories(yopo_maps PUBLIC include ${PCL_INCLUDE_DIRS})
target_link_libraries(yopo_maps PUBLIC Eigen3::Eigen ${PCL_LIBRARIES} yaml-cpp)
target_compile_options(yopo_maps PUBLIC ${PCL_DEFINITIONS})
add_executable(map_generator src/map_generator.cpp)
target_link_libraries(map_generator PRIVATE yopo_maps)

option(YOPO_BUILD_CUDA "Build the official CUDA image collector" ON)
if(YOPO_BUILD_CUDA)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(raycast_cuda STATIC
    src/sensor_simulator.cu
    src/simsense/core.cu src/simsense/csct.cu src/simsense/cost.cu
    src/simsense/aggr.cu src/simsense/wta.cu src/simsense/lrcheck.cu
    src/simsense/filter.cu src/simsense/camera.cu)
target_include_directories(raycast_cuda PUBLIC include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(raycast_cuda PUBLIC Eigen3::Eigen CUDA::cudart CUDA::curand ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
target_compile_options(raycast_cuda PUBLIC ${PCL_DEFINITIONS})

add_executable(dataset_generator src/dataset_generator.cpp)
target_link_libraries(dataset_generator PRIVATE raycast_cuda yopo_maps)
endif()
