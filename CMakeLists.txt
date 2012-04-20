# -*- cmake -*-

# support alternative targets
MESSAGE(STATUS "Targetting ARM6 architecture")

set(IPHONE_SDK "5.1")
set(IPHONE_ROOT "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer")
set(IPHONE_SDK_ROOT "${IPHONE_ROOT}/SDKs/iPhoneOS${IPHONE_SDK}.sdk")

set(CMAKE_FIND_ROOT_PATH "${IPHONE_SDK_ROOT}" "${IPHONE_ROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_SYSTEM_NAME "GNU")
set(CMAKE_SYSTEM_PROCESSOR armv6)
set(CMAKE_OSX_ARCHITECTURES armv6)

set(CMAKE_C_COMPILER ${IPHONE_ROOT}/usr/bin/gcc)
set(CMAKE_CXX_COMPILER ${IPHONE_ROOT}/usr/bin/g++)

set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

add_definitions("-D__IPHONE__")
add_definitions("-arch armv6 -pipe -no-cpp-precomp --sysroot=${IPHONE_SDK_ROOT} -miphoneos-version-min=${IPHONE_SDK}")
set(CMAKE_C_LINK_FLAGS "-arch armv6 --isysroot=${IPHONE_SDK_ROOT} -miphoneos-version-min=${IPHONE_SDK} -L${IPHONE_SDK_ROOT}/usr/lib -L${IPHONE_SDK_ROOT}/usr/lib/system")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_C_LINK_FLAGS})

include_directories("${IPHONE_SDK_ROOT}/usr/include")
include_directories("${IPHONE_SDK_ROOT}/usr/include/c++/4.2.1")

link_directories("${IPHONE_SDK_ROOT}/usr/lib")
link_directories("${IPHONE_SDK_ROOT}/usr/lib/system")

set(CMAKE_CROSSCOMPILING TRUE)
set(IPHONE TRUE)

set(OCIO_BUILD_SHARED FALSE)
set(OCIO_BUILD_STATIC TRUE)
set(OCIO_BUILD_TRUELIGHT FALSE)
set(OCIO_BUILD_TESTBED FALSE)
set(OCIO_BUILD_APPS FALSE)
set(OCIO_BUILD_NUKE FALSE)
set(OCIO_BUILD_PYGLUE FALSE)
set(OCIO_BUILD_SSE FALSE)

SET( C_HAS_inline_EXITCODE 
     "PLEASE_FILL_OUT-FAILED_TO_RUN"
     CACHE STRING "Result from TRY_RUN" FORCE)

# Project
PROJECT(Torch)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake"
                      "${CMAKE_BINARY_DIR}/cmake" "${CMAKE_MODULE_PATH}")

INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchCFlags)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)

# Torch executables
ADD_SUBDIRECTORY(exe)

# Torch libraries
ADD_SUBDIRECTORY(lib)

# Torch base packages
ADD_SUBDIRECTORY(pkg)

# Torch extra packages
ADD_SUBDIRECTORY(extra)

# External packages support
INCLUDE(TorchExports)

# Packaging support
INCLUDE(TorchCPack)
