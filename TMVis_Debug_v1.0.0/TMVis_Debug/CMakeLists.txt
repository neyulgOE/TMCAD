cmake_minimum_required(VERSION 3.10)

get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(TMVis_Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development Required NumPy)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/*.cpp"
    "${CMAKE_SOURCE_DIR}/ClientAPI/*.cpp"
    "${CMAKE_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_SOURCE_DIR}/GetData/*.cpp"
    "${CMAKE_SOURCE_DIR}/MarkerDetection/*.cpp"
    "${CMAKE_SOURCE_DIR}/CadPose/*.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ClientAPI"
    "${CMAKE_SOURCE_DIR}/common"
    "${CMAKE_SOURCE_DIR}/GetData"
    "${CMAKE_SOURCE_DIR}/MarkerDetection"
    "${CMAKE_SOURCE_DIR}/CadPose"
    "${CMAKE_SOURCE_DIR}/rapidjson"
    "${CMAKE_SOURCE_DIR}/ort_include"
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
    ${GLOG_LIBRARY}
    ${OpenCV_LIBS}
    onnxruntime
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
    INSTALL_RPATH "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
)

if (EXISTS "${CMAKE_SOURCE_DIR}/build/test_estimate.cpp")
    set(CAD_SOURCES
        "${CMAKE_SOURCE_DIR}/CadPose/CADSceneSegmenter.cpp"
        "${CMAKE_SOURCE_DIR}/CadPose/TablePoseEstimator.cpp"
        "${CMAKE_SOURCE_DIR}/CadPose/CADTemplateDetector.cpp"
    )
    add_executable(test_estimate "${CMAKE_SOURCE_DIR}/build/test_estimate.cpp" ${CAD_SOURCES})
    target_include_directories(test_estimate PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/CadPose"
        "${CMAKE_SOURCE_DIR}/GetData"
        "${CMAKE_SOURCE_DIR}/common"
        "${CMAKE_SOURCE_DIR}/rapidjson"
        "${CMAKE_SOURCE_DIR}/ort_include"
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(test_estimate PRIVATE
        "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
    )
    target_link_libraries(test_estimate PRIVATE
        ${Python3_LIBRARIES}
        ${OpenCV_LIBS}
        onnxruntime
    )
    set_target_properties(test_estimate PROPERTIES
        BUILD_RPATH "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
        INSTALL_RPATH "/home/huyen1905/.local/lib/python3.10/site-packages/onnxruntime/capi"
    )
endif()