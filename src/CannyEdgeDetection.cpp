/**********************************************************************************************************************
* File name: Canny Edge Detection.cpp
* 
* This program implements Canny Edge detection on the input image.
* Histogram Equilization is applied as a pre-precessing step to equalize the image intensities throughout the image.
* This image is then given to the Canny edge detector.
*
* Canny edge detection algorithm has 5 steps:
* 1. Apply Gaussian filter
* 2. Apply Sobel filter
* 3. Apply non-maximum suppression
* 4. Apply a double threshold
* 5. Track the edges using hysteresis
* 
* This algorithm is implemented on both CPU and GPU and the performance speedup is displayed to the user.
* The output images are stored in the out\build\x64-Debug directory.
* 
* Project Team:
* Gopika Rajan (3575765)
* M.Mehroz Khan (3523539)
* Swathi Shridhar (3578034)
***********************************************************************************************************************
*/

/**********************************************************************************************************************
* Header files
***********************************************************************************************************************
*/
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include<string.h>

#include <boost/lexical_cast.hpp>
#include "Canny.h"


#define M_PI acos(-1.0)
#define NUMBER_OF_BINS 256

/**********************************************************************************************************************
* Function definitions
***********************************************************************************************************************
*/

/**********************************************************************************************************************
 * Function name: Performance
 * Measure and print performance data for different functionalities.
 * Parameters:
 *  event1_1 - Event corresponding to the memory copy of Input1 to the device
 *  event1_2 - Event corresponding to the memory copy of Input2 to the device
 *  event2 - Event corresponding to the kernel launch of a GPU functionality
 *  event3 - Event corresponding to the memory copy of output1 back to host
 *  event4 - Event corresponding to the memory copy of output2 back to host
 *  f - A string describing the functionality being measured.
 *  cputime - CPU time for the functionality.
 **********************************************************************************************************************
 */
void Performance(cl::Event* event1_1, cl::Event* event2, cl::Event* event3, cl::Event* event4, std::string f, Core::TimeSpan cputime, cl::Event* event1_2)
{
	Core::TimeSpan gpuExecutionTime = OpenCL::getElapsedTime(*event2); //gputime before memory copy
	Core::TimeSpan gpuMemCopyTime = Core::TimeSpan::fromSeconds(0);  //gputime after memory copy

	if (event4 == nullptr)
	{
		if (event1_2 == nullptr)
		{
			gpuMemCopyTime = OpenCL::getElapsedTime(*event1_1) + OpenCL::getElapsedTime(*event3);
		}
		if (event1_2 != nullptr)
		{
			gpuMemCopyTime = OpenCL::getElapsedTime(*event1_1) + OpenCL::getElapsedTime(*event1_2) + OpenCL::getElapsedTime(*event3);
		}
	}
	else
	{
		gpuMemCopyTime = OpenCL::getElapsedTime(*event1_1) + OpenCL::getElapsedTime(*event3) + OpenCL::getElapsedTime(*event4);
	}
	Core::TimeSpan totalGpuTime = gpuExecutionTime + gpuMemCopyTime; //total gpu time

	//String stream to format and print the performance data
	std::stringstream str;
	str << std::setiosflags(std::ios::left) << std::setw(20) << f;
	str << std::setiosflags(std::ios::right);
	str << " " << std::setw(10) << cputime.toString();
	str << " " << std::setw(12) << gpuExecutionTime.toString();
	str << " " << std::setw(15) << totalGpuTime.toString();
	str << " " << std::setw(14) << (cputime.getSeconds() / gpuExecutionTime.getSeconds());
	str << " " << std::setw(15) << (cputime.getSeconds() / totalGpuTime.getSeconds());
	std::cout << str.str() << std::endl;
}

/**********************************************************************************************************************
 * Overloaded function to measure and print performance data for different functionalities with two input copy and 
 * only single output copy back event.
 * Parameters:
 *  event1_1 - Event corresponding to the memory copy of Input1 to the device
 *  event1_2 - Event corresponding to the memory copy of Input2 to the device
 *  event2 - Event corresponding to the kernal launch of a GPU functionality
 *  event3 - Event corresponding to the memory copy of output1 back to host
 *  f - A string describing the functionality being measured.
 *  cputime - CPU time for the functionality.
 **********************************************************************************************************************
 */
void Performance(cl::Event* event1_1, cl::Event* event2, cl::Event* event3, std::string f, Core::TimeSpan cputime, cl::Event* event1_2)
{
	// Call the main gputime function with event4 set to nullptr.
	Performance(event1_1, event2, event3, nullptr, f, cputime, event1_2);
}

/**********************************************************************************************************************
 * Overloaded function to measure and print performance data for different functionalities with only single input copy
 * and two output copy back events.
 * Parameters:
 *  event1_1 - Event corresponding to the memory copy of Input1 to the device
 *  event2 - Event corresponding to the kernal launch of a GPU functionality
 *  event3 - Event corresponding to the memory copy of output1 back to host
 *  event4 - Event corresponding to the memory copy of output2 back to host
 *  f - A string describing the functionality being measured.
 *  cputime - CPU time for the functionality.
 **********************************************************************************************************************
 */
void Performance(cl::Event* event1_1, cl::Event* event2, cl::Event* event3, cl::Event* event4, std::string f, Core::TimeSpan cputime)
{
	// Call the main gputime function with event1_2 set to nullptr.
	Performance(event1_1, event2, event3, event4, f, cputime, nullptr);
}

/**********************************************************************************************************************
 * Overloaded function to measure and print performance data for different functionalities with only single
 * input copy and output copy back events.
 * Parameters:
 *  event1_1 - Event corresponding to the memory copy of Input1 to the device
 *  event2 - Event corresponding to the kernal launch of a GPU functionality
 *  event3 - Event corresponding to the memory copy of output1 back to host
 *  event4 - Event corresponding to the memory copy of output2 back to host
 *  f - A string describing the functionality being measured.
 *  cputime - CPU time for the functionality.
 **********************************************************************************************************************
 */
void Performance(cl::Event* event1_1, cl::Event* event2, cl::Event* event3, std::string f, Core::TimeSpan cputime)
{
	// Call the main gputime function with event 1_2, event4 set to nullptr.
	Performance(event1_1, event2, event3, nullptr, f, cputime, nullptr);
}

/******************************************************************************************************************************
* Function name: main
* This is the main function. In this function the input image is imported and the histogram equalization and canny edge detection
* algorithms are applied using CPU and GPU. The performance of these two implementations are compared and displayed on the console,
* and the output images are stored.
* Parameters:
*  none
* Return:
*  int
********************************************************************************************************************************
*/
int main(int argc, char** argv) 
{
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	
	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/CannyEdgeDetection.cl");

	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Use an image as input data: Other input images are available in the src/InputImages folder.
	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
	Core::readImagePGM("../../../src/InputImages/Lizard.pgm", inputData, inputWidth, inputHeight);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * (inputWidth/16); // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * (inputHeight/16);
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu_HistogramEqualization(count);	
	std::vector<float> h_outputCpu_Canny(count);

	std::vector<int> h_outputGpu_HistogramCalculation(NUMBER_OF_BINS);
	std::vector<int> cdf(NUMBER_OF_BINS);
	std::vector<float> h_outputGpu_HistogramEqualization(count);
	std::vector<float> h_outputGpu_Gaussian(count);
	std::vector<float> h_outputGpu_Sobel(count);
	std::vector<int> h_out_segmentGpu(count);
	std::vector<float> h_outputGpu_NonMaxSupression(count);
	std::vector<float> h_outputGpu_Doublethreshold(count);
	std::vector<float> h_outputGpu_Canny(count);

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output_HistogramCalculation(context, CL_MEM_READ_WRITE, NUMBER_OF_BINS* sizeof(int));
	cl::Buffer d_input_HistogramEqualization(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output_HistogramEqualization(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputGpu_Gaussian(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputGpu_Gaussian(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputGpu_Sobel(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputGpu_Sobel(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_out_segment(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputGpu_NonMaxSupression(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputGpu_NonMaxSupression(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_in_segment(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputCpu_Sobel(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputDt(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputDt(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputHst(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputHst(context, CL_MEM_READ_WRITE, size);
	
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu_Canny.data(), 255, size);

	memset(h_outputGpu_Sobel.data(), 255, size);
	memset(h_outputGpu_HistogramCalculation.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(cdf.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(h_outputGpu_HistogramEqualization.data(), 255, size);
	memset(h_outputGpu_Gaussian.data(), 255, size);
	memset(h_outputGpu_NonMaxSupression.data(), 255, size);
	memset(h_outputGpu_Doublethreshold.data(), 255, size);
	memset(h_outputGpu_Canny.data(), 255, size);
	memset(h_out_segmentGpu.data(), 255, size);
	
	//Load input data into h_input
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {

			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)]; //not normalized

			if (h_input[i + countX * j] < 0)
			{
				h_input[i + countX * j] = 0;
			}
			else if (h_input[i + countX * j] > 255)
			{
				h_input[i + countX * j] = 255;

			}
		}
	}
		
	/******************************************Calculation on the host side*******************************************/
	histogramEqualization(h_outputCpu_HistogramEqualization, h_input, countX, countY);
	applyCanny_CPU(h_outputCpu_Canny, h_outputCpu_HistogramEqualization, countX, countY, count, size);
	/*****************************************************************************************************************/
	
	// Reinitialize output memory to 0xff
	memset(h_outputGpu_Sobel.data(), 255, size);
	memset(cdf.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(h_outputGpu_HistogramEqualization.data(), 255, size);
	memset(h_outputGpu_Gaussian.data(), 255, size);
	memset(h_outputGpu_NonMaxSupression.data(), 255, size);
	memset(h_out_segmentGpu.data(), 255, size);
	memset(h_outputGpu_Doublethreshold.data(), 255, size);
	memset(h_outputGpu_Canny.data(), 255, size);
	
	/******************************************Calculation on the GPU*************************************************/
	
	/*----------------------------------------- Histogram calculation-----------------------------------------------*/
	
	// Copy input data to device
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, NULL);

	//Create a kernel object
	cl::Kernel calculateHistogramKernel(program, "calculateHistogramKernel");

	//Launch kernel
	calculateHistogramKernel.setArg<cl::Buffer>(0, d_input);
	calculateHistogramKernel.setArg<cl::Buffer>(1, d_output_HistogramCalculation);
	queue.enqueueNDRangeKernel(calculateHistogramKernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

	//Copy output data back to the host
	queue.enqueueReadBuffer(d_output_HistogramCalculation, true, 0, NUMBER_OF_BINS * sizeof(int), h_outputGpu_HistogramCalculation.data(), NULL, NULL);

	cdf[0] = h_outputGpu_HistogramCalculation[0];
	for (int i = 1; i < NUMBER_OF_BINS; i++)
	{
		cdf[i] = cdf[i - 1] + h_outputGpu_HistogramCalculation[i];
	}

	/*----------------------------------------- Histogram Equalization-----------------------------------------------*/
	
	// Copy input data to device
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, NULL);
	queue.enqueueWriteBuffer(d_output_HistogramCalculation, true, 0, NUMBER_OF_BINS * sizeof(int), cdf.data(), NULL, NULL);
	
	//Create a kernel object
	cl::Kernel histogramEqualizationKernel(program, "histogramEqualizationKernel");

	//Launch kernel
	histogramEqualizationKernel.setArg<cl::Buffer>(0, d_input);
	histogramEqualizationKernel.setArg<cl::Buffer>(1, d_output_HistogramCalculation);
	histogramEqualizationKernel.setArg<cl::Buffer>(2, d_output_HistogramEqualization);
	queue.enqueueNDRangeKernel(histogramEqualizationKernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

	//Copy output data back to the host
	queue.enqueueReadBuffer(d_output_HistogramEqualization, true, 0, size, h_outputGpu_HistogramEqualization.data(), NULL, NULL);

	/*------------------------------------------------- 1. Apply Gaussian filter----------------------------------------*/
	cl::Event eventG1;
	// Copy input data to device
	queue.enqueueWriteBuffer(d_inputGpu_Gaussian, true, 0, size, h_outputGpu_HistogramEqualization.data(), NULL, &eventG1);

	//Create a kernel object
	cl::Kernel gaussianKernel(program, "gaussianKernel");

	//Launch kernel
	gaussianKernel.setArg<cl::Buffer>(0, d_inputGpu_Gaussian);
	gaussianKernel.setArg<cl::Buffer>(1, d_outputGpu_Gaussian);
	cl::Event eventG2;
	queue.enqueueNDRangeKernel(gaussianKernel, 
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		&eventG2);

	//Copy output data back to the host
	cl::Event eventG3;
	queue.enqueueReadBuffer(d_outputGpu_Gaussian, true, 0, size, h_outputGpu_Gaussian.data(), NULL, &eventG3);
	
	/*------------------------------------------------- 2. Apply Sobel filter----------------------------------------*/
	cl::Event eventS1;
	// Copy input data to device
	queue.enqueueWriteBuffer(d_inputGpu_Sobel, true, 0, size, h_outputGpu_Gaussian.data(), NULL, &eventS1);

	// Create a kernel object
	cl::Kernel sobelKernel(program, "sobelKernel");

	// Launch kernel on the device
	sobelKernel.setArg<cl::Buffer>(0, d_inputGpu_Sobel);
	sobelKernel.setArg<cl::Buffer>(1, d_outputGpu_Sobel);
	sobelKernel.setArg<cl::Buffer>(2, d_out_segment);
	cl::Event eventS2;
	queue.enqueueNDRangeKernel(sobelKernel,
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		&eventS2);

	// Copy output data back to host
	cl::Event eventS3;
	queue.enqueueReadBuffer(d_outputGpu_Sobel, true, 0, size, h_outputGpu_Sobel.data(), NULL, &eventS3);
	cl::Event eventS4;
	queue.enqueueReadBuffer(d_out_segment, true, 0, size, h_out_segmentGpu.data(), NULL, &eventS4);
	
	/*--------------------------------------------3. Apply Non Max Supression----------------------------------------*/

	cl::Event eventNM1;
	cl::Event eventNM2;
	// Copy input data to device
	queue.enqueueWriteBuffer(d_inputGpu_NonMaxSupression, true, 0, size, h_outputGpu_Sobel.data(), NULL, &eventNM1);
	queue.enqueueWriteBuffer(d_in_segment, true, 0, size, h_out_segmentGpu.data(), NULL, &eventNM2);

	// Create a kernel object
	cl::Kernel nonMaxSuppressionKernel(program, "nonMaxSuppressionKernel");

	// Launch kernel on the device
	nonMaxSuppressionKernel.setArg<cl::Buffer>(0, d_inputGpu_NonMaxSupression);
	nonMaxSuppressionKernel.setArg<cl::Buffer>(1, d_outputGpu_NonMaxSupression);
	nonMaxSuppressionKernel.setArg<cl::Buffer>(2, d_in_segment);
	cl::Event eventNM3;
	queue.enqueueNDRangeKernel(nonMaxSuppressionKernel,
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		&eventNM3);

	// Copy output data back to host
	cl::Event eventNM4;
	queue.enqueueReadBuffer(d_outputGpu_NonMaxSupression, true, 0, size, h_outputGpu_NonMaxSupression.data(), NULL, &eventNM4);
	
	/*------------------------------------------------ 4. Apply Double threshold----------------------------------------*/
	cl::Event eventDt1;
	// Copy input data to device
	queue.enqueueWriteBuffer(d_inputDt, true, 0, size, h_outputGpu_NonMaxSupression.data(), NULL, &eventDt1);

	// Create a kernel object
	cl::Kernel DoubleThresholdKernel(program, "DoubleThresholdKernel");

	// Launch kernel on the device
	DoubleThresholdKernel.setArg<cl::Buffer>(0, d_inputDt);
	DoubleThresholdKernel.setArg<cl::Buffer>(1, d_outputDt);
	DoubleThresholdKernel.setArg<cl_float>(2, low_threshold);
	DoubleThresholdKernel.setArg<cl_float>(3, high_threshold);
	cl::Event eventDt2;
	queue.enqueueNDRangeKernel(DoubleThresholdKernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &eventDt2);
	
	// Copy output data back to host
	cl::Event eventDt3;
	queue.enqueueReadBuffer(d_outputDt, true, 0, size, h_outputGpu_Doublethreshold.data(), NULL, &eventDt3);

	/*------------------------------------------------- 5. Apply Edge Hysterisis----------------------------------------*/
	cl::Event eventHst1;
	// Copy input data to device
	queue.enqueueWriteBuffer(d_inputHst, true, 0, size, h_outputGpu_Doublethreshold.data(), NULL, &eventHst1);

	// Create a kernel object
	cl::Kernel HysteresisKernel(program, "HysteresisKernel");

	// Launch kernel on the device
	HysteresisKernel.setArg<cl::Buffer>(0, d_inputHst);
	HysteresisKernel.setArg<cl::Buffer>(1, d_outputHst);

	cl::Event eventHst2;
	queue.enqueueNDRangeKernel(HysteresisKernel,cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &eventHst2);

	// Copy output data back to host
	cl::Event eventHst3;
	queue.enqueueReadBuffer(d_outputHst, true, 0, size, h_outputGpu_Canny.data(), NULL, &eventHst3);
	
	/************************************** Calculating cpu time for different functionalities************************/

	Core::TimeSpan cputimeGaussian = cpuendGaussian - cpubeginGaussian;
	Core::TimeSpan cputimeSobel = cpuendSobel - cpubeginsobel;
	Core::TimeSpan cputimeNonmaxsupression = cpuendNonmaxsuppression - cpuendSobel;
	Core::TimeSpan cputimeDoublethreshold = cpuendDoublethreshold - cpuendNonmaxsuppression;
	Core::TimeSpan cputimeHysteresis = cpuendHysteresis - cpuendDoublethreshold;
	Core::TimeSpan cputimeCanny = cpuendHysteresis - cpubeginGaussian;

	/************************************************* Print Performance data ****************************************/
	
	std::cout << "************************************  CANNY EDGE DETECTOR  ***********************************" << std::endl;
	std::cout << std::endl << "Device information:" << std::endl;	
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	OpenCL::printDeviceInfo(std::cout, device);

	std::cout << std::endl << "Input Image information:" << std::endl;
	std::cout << "Input Image Resolution: " << countX << " x " << countY << " Pixels " << std::endl;
	std::cout << "Input Image format: Grayscale (.pgm)" << std::endl;

	std::cout << std::endl << "Performance data for implementation :" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    
	/* String stream for performance headers*/
	std::stringstream str1;
	str1 << std::setiosflags(std::ios::left) << std::setw(20) << "Functionality";
	str1 << std::setiosflags(std::ios::right);
	str1 << " " << std::setw(9) << "| CpuTime |";
	str1 << " " << std::setw(9) << "GpuTime w/o MC |";
	str1 << " " << std::setw(9) << "TotalGpuTime |";
	str1 << " " << std::setw(9) << "Speedup w/o MC |";
	str1 << " " << std::setw(9) << " Speedup MC  |";
	std::cout << str1.str() << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
   
	Performance(&eventG1, &eventG2, &eventG3, "Gaussian", cputimeGaussian);
	Performance(&eventS1, &eventS2, &eventS3, &eventS4, "Sobel", cputimeSobel);
	Performance(&eventNM1, &eventNM3, &eventNM4, "NonMaxSupression", cputimeNonmaxsupression, &eventNM2);
	Performance(&eventDt1, &eventDt2, &eventDt3, "DoubleThreshold", cputimeDoublethreshold);
	Performance(&eventHst1, &eventHst2, &eventHst3, "Hysteresis", cputimeHysteresis);
	
	Core::TimeSpan gputime_canny = OpenCL::getElapsedTime(eventG2) + OpenCL::getElapsedTime(eventS2) 
								 + OpenCL::getElapsedTime(eventNM3) + OpenCL::getElapsedTime(eventDt2) 
								 + OpenCL::getElapsedTime(eventHst2);

	Core::TimeSpan gputime_canny_total = OpenCL::getElapsedTime(eventG1) + OpenCL::getElapsedTime(eventS1) 
										+ OpenCL::getElapsedTime(eventNM1) + OpenCL::getElapsedTime(eventNM2) + OpenCL::getElapsedTime(eventDt1) 
										+ OpenCL::getElapsedTime(eventHst1) + OpenCL::getElapsedTime(eventG3) + OpenCL::getElapsedTime(eventS3) 
										+ OpenCL::getElapsedTime(eventS4) + OpenCL::getElapsedTime(eventNM4) 
										+ OpenCL::getElapsedTime(eventDt3) + OpenCL::getElapsedTime(eventHst3) + gputime_canny;
	std::stringstream str2;
	std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
	str2 << std::setiosflags(std::ios::left) << std::setw(20) << "CannyEdgeDetection";
	str2 << std::setiosflags(std::ios::right);
	str2 << " " << std::setw(10) << cputimeCanny.toString();
	str2 << " " << std::setw(12) << gputime_canny.toString();
	str2 << " " << std::setw(15) << gputime_canny_total.toString();
	str2 << " " << std::setw(14) << (cputimeCanny.getSeconds() / gputime_canny.getSeconds());
	str2 << " " << std::setw(15) << (cputimeCanny.getSeconds() / gputime_canny_total.getSeconds());
	std::cout << str2.str() << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    
	/* Store GPU output image */
	Core::writeImagePGM("7_Histogram_Equalization_Gpu_Output.pgm", h_outputGpu_HistogramEqualization, countX, countY);
	Core::writeImagePGM("8_Gaussian_Gpu_Output.pgm", h_outputGpu_Gaussian, countX, countY);
	Core::writeImagePGM("9_Sobel_Gpu_Output.pgm", h_outputGpu_Sobel, countX, countY);
	Core::writeImagePGM("10_NonMaxSupression_Gpu_Output.pgm", h_outputGpu_NonMaxSupression, countX, countY);
	Core::writeImagePGM("11_DoubleThreshold_Gpu_Output.pgm", h_outputGpu_Doublethreshold, countX, countY);
	Core::writeImagePGM("12_CannyEdgeDetection_Gpu_Output.pgm", h_outputGpu_Canny, countX, countY);	
	
	/* Error between CPU and GPU outputs */
	std::size_t errorCount = 0;
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			size_t index = i + j * countX;
			if (!(std::abs(h_outputGpu_Canny[index] - h_outputCpu_Canny[index]) == 0)) {
				errorCount++;
			}
		}
	}
	/* Print accuracy */
	float accuracy = (1- (float) errorCount/ (float)(countX * countY)) *100 ;
	//accuracy = accuracy * 100;
	std::cout << "Accuracy of GPU output compared to CPU output = " << accuracy << "%" << std::endl;
	std::cout << std::endl;
	std::cout << "Success" << std::endl;
	
	return 0;
}

