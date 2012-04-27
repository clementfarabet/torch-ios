CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(src opencv.c)
SET(luasrc init.lua img1.jpg img2.jpg)

ADD_TORCH_PACKAGE(opencv "${src}" "${luasrc}" "Image Processing")
TARGET_LINK_LIBRARIES(opencv luaT TH ${OpenCV_LIBS})
