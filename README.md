# Canny_edge_detection_opencl

This is the CPU and GPU(based on Opencl) implementation of canny edge detection for High Performance Computing with Graphic Cards praktikum.
## Steps in Canny_edge_detection
Canny edge detection consists of 5 steps.
1. Gaussian Filter
2. Sobel Filter
3. Non-Max Suppression
4. Double Thresold
5. Hysteresis edge tracking

--TODO-- (Add a collage of images with names of step in one row)

Here is how you can use this code to determine edges in the images and the intermediate results.  

Instructions to run this code in Visual Studio

### Environment Configuration:
1. Download NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Download Boost library from https://sourceforge.net/projects/boost/files/boost-binaries/1.76.0_b1/boost_1_76_0_b1-msvc-14.2-64.exe

### Project Configuration:
1. Download the Canny_edge_detection_opencl project from the github. 
2. In the CMakeLists.txt change the boost include directory and boost library directory to system include folder.
      - set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
      - set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2")

### Building the project
1.  Open Visual Studio-> Choose "Open a local folder" -> and select "Canny_edge_detection_opencl" folder.  
2.  CMake generation will automatically start.
3.  Build the project (Ctrl + Shift + B)
4.  In the solution explorer
      Choose "Switch between solutions and the available views"
      Choose "CMake Targets View"
      Expand "CannyEdgeDetection Project"
      Configure the "CannyEdgeDetection (executable)" as a startup item.
### Execution
  1. Execution is started by clicking "DisparityMap.exe" play button.
  2. For each of these [steps](https://github.com/Mehrozkhan/Canny_edge_detection_opencl/blob/mehroz/README.md#steps-in-canny_edge_detection), a CPU and GPU output image will be generated and saved in out/build/x64-Debug.

### Executing different example
1. Differnt sample examples are placed in the images directory (src/InputImages).
2. To run differnt example set we have to change the image name in the CannyEdgeDetection.cpp source file(Provide the complete path for images).
      - --TODO-- (code line to change)
3. Rebuid the project and execute.

