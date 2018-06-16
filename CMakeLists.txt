include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

cmake_minimum_required(VERSION 2.8)
project(project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)
add_executable(project src/main.cpp src/utils.cpp src/person_detector.cpp)
target_link_libraries(project ${OpenCV_LIBS})

add_definitions(-std=c++11)
enable_cxx_compiler_flag_if_supported("-Wall")
