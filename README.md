# Canny_edge_detection_opencl

This is the CPU and GPU(based on Opencl) implementation of canny edge detection for High Performance Computing with Graphic Cards praktikum.
Instructions to run this code in Visual Studio

Here is how you can use this code to determine edges in the images and the intermediate results.  --TODO-- (Add a collage of images with names of step in one row)

Instructions to run this code in Visual Studio

### Environment Configuration:
1. Download NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Download Boost library from https://sourceforge.net/projects/boost/files/boost-binaries/1.76.0_b1/boost_1_76_0_b1-msvc-14.2-64.exe

### Project Configuration:
1. Download the GPU_LAB_UniStuttgart_SS2022 project from the github. --TODO-- (Name of final repo)
2. In the CMakeLists.txt change the boost include directory and boost library directory to system include folder.
      - set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
      - set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2")

### Building the project
1.  Open Visual Studio-> Choose "Open a local folder" -> and select "GPU_LAB_UniStuttgart_SS2022" folder.  --TODO-- (Name of final repo)
2.  CMake generation will automatically start.
3.  Build the project (Ctrl + Shift + B)
   --TODO--
### Execution
  --TODO--
### Handling dependencies at run time.
### Executing different example
### Modification of the code

