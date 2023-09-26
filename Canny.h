#pragma once
extern float high_threshold;
extern float low_threshold;

extern Core::TimeSpan cpubeginGaussian;
extern Core::TimeSpan cpuendGaussian;
extern Core::TimeSpan cpubeginsobel;
extern Core::TimeSpan cpuendSobel;
extern Core::TimeSpan cpuendNonmaxsuppression;
extern Core::TimeSpan cpuendDoublethreshold;
extern Core::TimeSpan cpuendHysteresis;

int getIndexGlobal(std::size_t countX, int i, int j);
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j);
void calculateHistogram(std::vector<int>& histogram, std::vector<float>& h_input, std::size_t countX, std::size_t countY);
void histogramEqualization(std::vector<float>& h_outputCpu, std::vector<float>& h_input, std::size_t countX, std::size_t countY);
void gaussianFilter(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY);
void sobelEdgeDetector(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_out_segment, std::size_t countX, std::size_t countY);
void nonMaxSuppression(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_in_segment, std::size_t countX, std::size_t countY);
void applyDoubleThreshold(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY);
void applyEdgeHysteresis(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY);
void applyCanny_CPU(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY, std::size_t count, std::size_t size);
void applyCanny_GPU(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY,
	std::size_t count, std::size_t size);