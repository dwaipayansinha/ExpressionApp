#include<iostream>
#include<fstream>
#include<cstdlib>

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;

namespace fs = ::boost::filesystem;

void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret);

//string cascadeName = "../data/lbpcascades/lbpcascade_frontalface.xml";

int main()
{
	vector<fs::path> img_files;
	vector <fs::path> emo_files;
	vector<fs::path>::iterator it_img;
	vector<fs::path>::iterator it_emo;

	vector<Mat> images;
	vector<int> labels;

	get_all("../data/Emotion", ".txt", emo_files);
	get_all("../data/cohn-kanade-images", ".png", img_files);

	/*CascadeClassifier cascade;
	if (!cascade.load(cascadeName)) {
	cerr << "ERROR: Could not load classifier cascade" << endl;
	return EXIT_FAILURE;
	}*/

	for (it_emo = emo_files.begin(); it_emo < emo_files.end(); it_emo++)
	{
		string filename = it_emo->stem().string().substr(0, 17);
		//cout << filename << endl;
		for (it_img = img_files.begin(); it_img < img_files.end(); it_img++)
		{
			string imname = it_img->stem().string();
			if (imname.compare(0, filename.length(), filename) == 0)
			{
				try
				{
					Mat img = imread(it_img->string());
					imshow("Training", img);
					cout << filename << " " << img.rows << "x" << img.cols << " ";
					Mat gray;
					cvtColor(img, gray, CV_BGR2GRAY);
					/*equalizeHist(gray, gray);
					vector<Rect> faces;
					cascade.detectMultiScale(gray, faces, 1.1, 3, 0);
					Mat face = gray(faces[0]);
					images.push_back(face);*/
					images.push_back(gray);
					ifstream emofile(it_emo->string().c_str());
					string line;
					getline(emofile, line);
					emofile.close();
					int emo = atoi(line.c_str());
					cout << emo << endl;
					labels.push_back(emo);
					waitKey(1);
				}
				catch (cv::Exception& e)
				{
					cerr << "Error opening file \"" << filename << "\". Reason: " << e.msg << endl;
					// nothing more we can do
					exit(EXIT_FAILURE);
				}
			}
		}
	}
	Ptr<face::FaceRecognizer> model = face::createLBPHFaceRecognizer();
	cout << "Loading complete. Training model..." << endl;
	model->train(images, labels);
	model->setLabelInfo(0, "Neutral");
	model->setLabelInfo(1, "Anger");
	model->setLabelInfo(2, "Contempt");
	model->setLabelInfo(3, "Disgust");
	model->setLabelInfo(4, "Fear");
	model->setLabelInfo(5, "Happiness");
	model->setLabelInfo(6, "Sadness");
	model->setLabelInfo(7, "Surprise");
	cout << "Saving model..." << endl;
	model->save("../data/emotion_model.yml");


	return EXIT_SUCCESS;
}

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories

void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret)
{
	if (!fs::exists(root) || !fs::is_directory(root))
		return;

	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;

	while (it != endit)
	{
		if (fs::is_regular_file(*it) && it->path().extension() == ext)
			ret.push_back(it->path());
		++it;
	}
}
