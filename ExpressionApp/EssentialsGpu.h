#pragma once
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "EssentialsCpu.h"

using namespace cv::cuda;

void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, Ptr<cuda::CascadeClassifier>& cascade, double scale = 1.0);
void classifyFaces(vector<Rect>& faces, Mat& img, GpuMat& gray_gpu, Ptr<face::FaceRecognizer>& model);