#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

using namespace cv;
using namespace std;

void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, CascadeClassifier& cascade, double scale = 1.0);

string face_cascade_name = "data/lbpcascades/lbpcascade_frontalface.xml";

int main()
{
	Ptr<face::FaceRecognizer> model = face::createLBPHFaceRecognizer();
	model->load("../ExpressionTrain/emotion_model.yml");

	VideoCapture capture;
	Mat frame;

	CascadeClassifier face_cascade;
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	};
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
			detectAndDraw(frame1, model, face_cascade);

			int c = waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	return EXIT_SUCCESS;
}

void detectAndDraw(Mat& img, Ptr<face::FaceRecognizer>& model, CascadeClassifier& cascade, double scale)
{
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] = {
		Scalar(255, 0, 0),
		Scalar(255, 128, 0),
		Scalar(255, 255, 0),
		Scalar(0, 255, 0),
		Scalar(0, 128, 255),
		Scalar(0, 255, 255),
		Scalar(0, 0, 255),
		Scalar(255, 0, 255)
	};
	Mat gray, smallImg;

	//imshow("Debug showing", img);
	//waitKey(1);
	//cout << "Debug 1 " << img.rows << "x" << img.cols << endl;

	cvtColor(img, gray, CV_BGR2GRAY);
	//cout << "Debug 2 " << gray.rows << "x" << gray.cols << endl;
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	//equalizeHist(smallImg, smallImg);

	//imshow("gray output", smallImg);
	//waitKey(1);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces);

	for (int i = 0; i < faces.size(); i++)
	{
		// Process face by face:
		Rect face_i = faces[i];
		//Rect face_i = faces[0];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = gray(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		//Mat face_resized;
		//cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		//int prediction = model->predict(face_resized);
		//int prediction=0;
		int prediction = model->predict(face);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(img, face_i, CV_RGB(0, 255, 0), 1);
		// Create the text we will annotate the box with:
		//string box_text = format("Prediction = %d", prediction);
		string box_text = format("Prediction = %s", model->getLabelInfo(prediction).c_str());
		//cout << prediction << " " << model->getLabelInfo(prediction) << endl;
		// Calculate the position for annotated text (make sure we don't
		// put illegal values in there):
		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		// And now put it into the image:
		putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
	}
	imshow("result", img);
}