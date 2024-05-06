cmake_minimum_required(VERSION 3.23)
project(ai_app)

set(APP_NAME app_host)

# **********************
# Disable in-source build.
# **********************
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source build is not allowed! Please specify a build folder.\n\tex:cmake -B build")
endif()

enable_language(CXX C ASM)

add_executable(${APP_NAME})

file(GLOB APP_SOURCES "${CMAKE_SOURCE_DIR}/src/*")

target_sources(${APP_NAME}
    PRIVATE ${APP_SOURCES}
)

set(APP_BUILD_FLAGS
    "-O3"
    "-g"
)

target_compile_features(${APP_NAME} PUBLIC cxx_std_14)
target_compile_options(${APP_NAME} PRIVATE ${APP_BUILD_FLAGS})

# *******************************************************
# Include AI Tools library and headers
# *******************************************************
# The following variables are defined for AI Tools library 
# and should be included when building the app.
# XMOS_AITOOLSLIB_DEFINITIONS
# XMOS_AITOOLSLIB_INCLUDES
# XMOS_AITOOLSLIB_LIBRARIES
#
if(NOT DEFINED ENV{XMOS_AITOOLSLIB_PATH})
    message(FATAL_ERROR "Path to XMOS AI Tools library and headers not set correctly!")
endif()
include($ENV{XMOS_AITOOLSLIB_PATH}/buildfiles/aitoolslib.cmake)

target_include_directories(${APP_NAME}
    PUBLIC ${XMOS_AITOOLSLIB_INCLUDES}
)
target_compile_definitions(${APP_NAME} PUBLIC
    ${APP_DEFINTIONS}
    ${XMOS_AITOOLSLIB_DEFINITIONS}
)
target_link_libraries(${APP_NAME} ${XMOS_AITOOLSLIB_LIBRARIES})

install(TARGETS ${APP_NAME} DESTINATION "${CMAKE_BINARY_DIR}/bin")
