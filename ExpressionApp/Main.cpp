#include <iostream>

#include "Essentials.h"

using namespace cv;
using namespace std;
using namespace cv::cuda;


string face_cascade_name = "../data/lbpcascades/lbpcascade_frontalface.xml";

int main()
{
	bool use_gpu = true;
	Ptr<face::FaceRecognizer> model = face::createLBPHFaceRecognizer();
	model->load("../data/emotion_model.yml");

	VideoCapture capture;
	Mat frame;

	if (!use_gpu)
	{
		cv::CascadeClassifier face_cascade_cpu;
		if (!face_cascade_cpu.load(face_cascade_name))
		{
			printf("--(!)Error loading\n");
			return -1;
		};
		//-- 1. Load the cascades

		//-- 2. Read the video stream
		if (!capture.open(0))
			cout << "Capture from camera didn't work" << endl;
		if (capture.isOpened())
		{
			while (true) {
				capture >> frame;
				if (frame.empty())
					break;

				Mat frame1 = frame.clone();
				detectAndDraw(frame1, model, face_cascade_cpu);

				int c = waitKey(10);
				if (c == 27 || c == 'q' || c == 'Q')
					break;
			}
		}
	}
	else
	{
		Ptr<cuda::CascadeClassifier> face_cascade_gpu = cuda::CascadeClassifier::create(face_cascade_name);
		if (!capture.open(0))
			cout << "Capture from camera didn't work" << endl;
		if (capture.isOpened())
		{
			while (true) {
				capture >> frame;
				if (frame.empty())
					break;

				Mat frame1 = frame.clone();
				detectAndDraw(frame1, model, face_cascade_gpu);

				int c = waitKey(10);
				if (c == 27 || c == 'q' || c == 'Q')
					break;
			}
		}
	}

	return EXIT_SUCCESS;
}