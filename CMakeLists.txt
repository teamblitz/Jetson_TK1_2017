cmake_minimum_required (VERSION 2.8)

project (track_targets)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_track_targets src/track_targets.cpp)

target_link_libraries(cv_track_targets ${OpenCV_LIBS})
