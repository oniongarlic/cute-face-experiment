cmake_minimum_required(VERSION 3.16)

project(CuteFace VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the packages via pkg-config
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# List of required packages
set(PKG_DEPENDENCIES libpq libmosquitto)

# Query pkg-config for the required packages
foreach(PKG ${PKG_DEPENDENCIES})
    pkg_check_modules(${PKG} REQUIRED ${PKG})
    message(STATUS "Found package: ${PKG}")
endforeach()

# Add executable source files
add_executable(cuteface
    main.cpp
    openface.cpp openface.hpp
    moving_average.cpp moving_average.hpp
    selfiesegment.cpp selfiesegment.hpp
    focus_check.cpp focus_check.hpp
    yolov8_face.cpp yolov8_face.h
)

# Include directories from pkg-config
target_include_directories(cuteface PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${libpq_INCLUDE_DIRS}
    ${libmosquitto_INCLUDE_DIRS}
)

# Link libraries from pkg-config
target_link_libraries(cuteface PRIVATE
    ${OpenCV_LIBS}
    ${libpq_LIBRARIES}
    ${libmosquitto_LIBRARIES}
)

# Add other necessary flags
target_compile_options(cuteface PRIVATE
    ${OpenCV_CFLAGS_OTHER}
    ${libpq_CFLAGS_OTHER}
    ${libmosquitto_CFLAGS_OTHER}
)

