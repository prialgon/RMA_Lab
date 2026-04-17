cmake_minimum_required(VERSION 3.16)
project(amr_localization_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Find nanobind installed via pip
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Build nanobind extension module
nanobind_add_module(_pf_core NOMINSIZE
  src/pf_core.cpp
)

target_include_directories(_pf_core PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)

# Put the extension inside the installed Python package
install(
  TARGETS _pf_core
  LIBRARY DESTINATION local/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages/${PROJECT_NAME}
)

# Install platform-specific C helper library next to maps.py
install(
  FILES
    ${PROJECT_NAME}/libintersect.so
    ${PROJECT_NAME}/libintersect.dylib
    ${PROJECT_NAME}/libintersect.dll
  DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/${PROJECT_NAME}
  OPTIONAL
)

install(
  DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    amr_localization_cpp/particle_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()