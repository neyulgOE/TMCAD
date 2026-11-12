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

get_filename_component(TMCAD_ROOT_DIR "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

set(ORT_PYTHON_CAPI_DIR "")
if(Python3_EXECUTABLE)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
        OUTPUT_VARIABLE ORT_PYTHON_CAPI_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

set(ORT_LINK_DIRS "${TMCAD_ROOT_DIR}/libs/onnxruntime")
if(ORT_PYTHON_CAPI_DIR AND EXISTS "${ORT_PYTHON_CAPI_DIR}")
    list(APPEND ORT_LINK_DIRS "${ORT_PYTHON_CAPI_DIR}")
endif()
list(JOIN ORT_LINK_DIRS ";" ORT_RPATH)

# Reuse Single-Source-of-Truth pose core from root src/
add_library(tmcad_pose_core STATIC
    "${TMCAD_ROOT_DIR}/src/CADSceneSegmenter.cpp"
    "${TMCAD_ROOT_DIR}/src/CADTemplateDetector.cpp"
    "${TMCAD_ROOT_DIR}/src/UniversalPoseEstimator.cpp"
)
target_include_directories(tmcad_pose_core PUBLIC
    "${TMCAD_ROOT_DIR}/src"
    "${TMCAD_ROOT_DIR}/libs/onnxruntime/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(tmcad_pose_core PUBLIC ${ORT_LINK_DIRS})
target_link_libraries(tmcad_pose_core PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
)

file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/*.cpp"
    "${CMAKE_SOURCE_DIR}/ClientAPI/*.cpp"
    "${CMAKE_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_SOURCE_DIR}/GetData/*.cpp"
    "${CMAKE_SOURCE_DIR}/MarkerDetection/*.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ClientAPI"
    "${CMAKE_SOURCE_DIR}/common"
    "${CMAKE_SOURCE_DIR}/GetData"
    "${CMAKE_SOURCE_DIR}/MarkerDetection"
    "${CMAKE_SOURCE_DIR}/rapidjson"
    "${TMCAD_ROOT_DIR}/src"
    "${TMCAD_ROOT_DIR}/libs/onnxruntime/include"
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE ${ORT_LINK_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    tmcad_pose_core
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
    ${GLOG_LIBRARY}
    ${OpenCV_LIBS}
    onnxruntime
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${ORT_RPATH}"
    INSTALL_RPATH "${ORT_RPATH}"
)

if (EXISTS "${CMAKE_SOURCE_DIR}/build/test_estimate.cpp")
    add_executable(test_estimate "${CMAKE_SOURCE_DIR}/build/test_estimate.cpp")
    target_include_directories(test_estimate PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/GetData"
        "${CMAKE_SOURCE_DIR}/common"
        "${CMAKE_SOURCE_DIR}/rapidjson"
        "${TMCAD_ROOT_DIR}/src"
        "${TMCAD_ROOT_DIR}/libs/onnxruntime/include"
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(test_estimate PRIVATE ${ORT_LINK_DIRS})
    target_link_libraries(test_estimate PRIVATE
        tmcad_pose_core
        ${Python3_LIBRARIES}
        ${OpenCV_LIBS}
        onnxruntime
    )
    set_target_properties(test_estimate PROPERTIES
        BUILD_RPATH "${ORT_RPATH}"
        INSTALL_RPATH "${ORT_RPATH}"
    )
endif()
