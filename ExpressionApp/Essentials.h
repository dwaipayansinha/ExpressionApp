#pragma once

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;
void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, cv::CascadeClassifier& cascade, double scale = 1.0);
void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, Ptr<cuda::CascadeClassifier>& cascade, double scale = 1.0);
void classifyFaces(vector<Rect>& faces, Mat& img, Mat& gray, Ptr<face::FaceRecognizer>& model);
void classifyFaces(vector<Rect>& faces, Mat& img, GpuMat& gray_gpu, Ptr<face::FaceRecognizer>& model);
