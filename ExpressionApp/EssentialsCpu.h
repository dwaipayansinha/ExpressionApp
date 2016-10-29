#pragma once

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

using namespace cv;
using namespace std;

void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, cv::CascadeClassifier& cascade, double scale = 1.0);
void classifyFaces(vector<Rect>& faces, Mat& img, Mat& gray, Ptr<face::FaceRecognizer>& model);