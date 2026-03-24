cmake_minimum_required(VERSION 3.8)
project(amr_localization_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(amr_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)

include_directories(include)

add_library(amr_localization_lib SHARED
  src/map.cpp
  src/particle_filter.cpp
)

target_include_directories(amr_localization_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(amr_localization_lib
  rclcpp
  ament_index_cpp
)

target_compile_features(amr_localization_lib PUBLIC cxx_std_17)

add_executable(particle_filter src/particle_filter_node.cpp)

target_link_libraries(particle_filter
  amr_localization_lib
)

ament_target_dependencies(particle_filter
  rclcpp
  rclcpp_lifecycle
  amr_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_geometry_msgs
  message_filters
  ament_index_cpp
)

target_compile_features(particle_filter PUBLIC cxx_std_17)

install(TARGETS
  amr_localization_lib
  particle_filter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY maps/
  DESTINATION share/${PROJECT_NAME}/maps
)

ament_package()