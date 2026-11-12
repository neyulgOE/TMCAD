cmake_minimum_required(VERSION 3.16)
project(CadViewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(Open3D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/cmake/Open3D")
find_package(Open3D REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(CadViewer
    src/main.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/Viewport3D.h
    src/Viewport3D.cpp
    src/SceneManager.h
    src/SceneManager.cpp
    src/SceneObject.h
    src/SceneObject.cpp
    src/CADAsset.h
    src/CADAsset.cpp
    src/CADHierarchy.h
    src/CADHierarchy.cpp
    src/CADPreprocessor.h
    src/CADPreprocessor.cpp
    src/PoseEstimator.h
    src/PoseEstimator.cpp
    src/RGBDSource.h
    src/RGBDSource.cpp
    src/RealSenseRecorderDialog.h
    src/RealSenseRecorderDialog.cpp
    src/CameraStreamWidget.h
    src/CameraStreamWidget.cpp
    src/SyntheticRGBDSource.cpp
    src/SyntheticDatasetGenerator.cpp
    src/CADTemplateDetector.h
    src/CADTemplateDetector.cpp
    src/ImageDetectionDialog.h
    src/ImageDetectionDialog.cpp
    src/CameraHardwareProfile.h
    src/CameraProfileParser.h
    src/CameraProfileParser.cpp
    src/MuJoCoD435SimSource.h
    src/MuJoCoD435SimSource.cpp
    src/CameraProfileDialog.h
    src/CameraProfileDialog.cpp
    src/MuJoCoSimDialog.h
    src/MuJoCoSimDialog.cpp
    src/CADSceneSegmenter.h
    src/CADSceneSegmenter.cpp
)

target_include_directories(CadViewer PRIVATE 
    ${Open3D_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco/include
)
target_link_libraries(CadViewer PRIVATE
    Qt5::Widgets
    Open3D::Open3D
    realsense2::realsense2
    ${OpenCV_LIBS}
    EGL
    GL
    glfw
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco/lib/libmujoco.so
)

add_executable(run_sweeps src/run_sweeps.cpp src/CADHierarchy.cpp src/CADPreprocessor.cpp src/CADAsset.cpp src/SceneManager.cpp src/SceneObject.cpp src/SyntheticRGBDSource.cpp src/PoseEstimator.cpp)
target_include_directories(run_sweeps PRIVATE ${Open3D_INCLUDE_DIRS})
target_link_libraries(run_sweeps PRIVATE Qt5::Widgets  PRIVATE
    Open3D::Open3D
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)

add_executable(generate_dataset src/generate_dataset.cpp src/CADHierarchy.cpp src/CADPreprocessor.cpp src/CADAsset.cpp src/SceneManager.cpp src/SceneObject.cpp src/SyntheticRGBDSource.cpp src/SyntheticDatasetGenerator.cpp)
target_include_directories(generate_dataset PRIVATE ${Open3D_INCLUDE_DIRS})
target_link_libraries(generate_dataset PRIVATE Qt5::Widgets PRIVATE
    Open3D::Open3D
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)

add_executable(run_benchmark 
    src/run_benchmark.cpp 
    src/CADHierarchy.cpp
    src/CADPreprocessor.cpp 
    src/CADAsset.cpp 
    src/SceneManager.cpp 
    src/SceneObject.cpp 
    src/RGBDSource.cpp 
    src/SyntheticRGBDSource.cpp 
    src/PoseEstimator.cpp
)
target_include_directories(run_benchmark PRIVATE ${Open3D_INCLUDE_DIRS})
target_link_libraries(run_benchmark PRIVATE Qt5::Widgets PRIVATE
    Open3D::Open3D
    realsense2::realsense2
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)

add_executable(test_phase0_phase1
    tests/test_phase0_phase1.cpp
    src/CADHierarchy.cpp
    src/CADPreprocessor.cpp
    src/CADAsset.cpp
)
target_include_directories(test_phase0_phase1 PRIVATE ${Open3D_INCLUDE_DIRS})
target_link_libraries(test_phase0_phase1 PRIVATE Qt5::Widgets PRIVATE
    Open3D::Open3D
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)

add_executable(record_realsense
    src/record_realsense.cpp
    src/RealSenseRecorderDialog.h
    src/RealSenseRecorderDialog.cpp
)
target_include_directories(record_realsense PRIVATE ${Open3D_INCLUDE_DIRS})
target_link_libraries(record_realsense PRIVATE
    Qt5::Widgets
    Open3D::Open3D
    realsense2::realsense2
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)

add_executable(test_mujoco_sim_source
    tests/test_mujoco_sim_source.cpp
    src/MuJoCoD435SimSource.h
    src/MuJoCoD435SimSource.cpp
    src/CameraProfileParser.h
    src/CameraProfileParser.cpp
)
target_include_directories(test_mujoco_sim_source PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Open3D_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco/include
)
target_link_libraries(test_mujoco_sim_source PRIVATE
    Qt5::Widgets
    Open3D::Open3D
    ${OpenCV_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco/lib/libmujoco.so
    glfw
    GL
    EGL
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++.so.1
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/open3d_cxx/lib/libc++abi.so.1
)
