// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

#include "data.h"

extern "C" {
#include<vl/kmeans.h>
#include<vl/dsift.h>
}


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

Point2i topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner;
int currentClick = 0;

void CallBackCorners(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{

		Point2i newPoint;
		newPoint.x = x;
		newPoint.y = y;

		switch (currentClick)
		{
			case 0:
				topLeftCorner = newPoint;
				break;
			case 1:
				topRightCorner = newPoint;
				break;
			case 2:
				bottomLeftCorner = newPoint;
				break;
			case 3:
				bottomRightCorner = newPoint;
				break;
			default:
				break;
		}
		currentClick++;
		if (currentClick == 4)
			;//launch detection method
	}
}

void shapeMouseClick()
{

//	boards_5_pieces.push_back(board_5_1);

	Mat_<Vec3b> src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", CallBackCorners, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


Mat_<uchar> rgb_to_gray(Mat_<Vec3b> img)
{
	Mat_<uchar> grayImg(img.rows, img.cols);

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			grayImg(i, j) = ((float)img(i, j)[0] / 4 + (float)img(i, j)[1] / 2 + (float)img(i, j)[2] / 4);
		}
	return grayImg;
}

float* gray_to_vector_normal(Mat_<uchar> gray_image)
{
	float* imgvec = (float *)malloc(gray_image.rows*gray_image.cols * sizeof(float));
	for (int i = 0; i < gray_image.rows; i++) {
		for (int j = 0; j < gray_image.cols; j++) {
			imgvec[i*gray_image.cols + j] = gray_image(i, j) / 255.0f;
		}
	}
	return imgvec;
}

int find_closest_center(Mat_<float> feature_vector, Mat_<float> centers)
{

	double min_dist = -1;
	double min_index = -1;

	for (int i = 0; i < 8; i++)
	{
		Mat_<float> current_center(1, 128);

		for (int j = 0; j < centers.cols; j++)
			current_center(0, j) = centers(i, j);
		double dist = norm(feature_vector, current_center, NORM_L2);

		if (dist < min_dist || min_dist == -1)
		{
			min_dist = dist;
			min_index = i;
		}
	}
	return min_index;
}

Mat_<float> get_histo(Mat_<float> current_features)
{
	int histo[8] = { 0,0,0,0,0,0,0,0 };

	for (int j = 0; j < current_features.rows; j++)
	{
		Mat_<float> feature_vector(1, 128);
		for (int q = 0; q < current_features.cols; q++)
			feature_vector(0, q) = current_features(j, q);

		histo[find_closest_center(feature_vector, centers)]++;
	}

	Mat_<float> histogram(1, 8);
	for (int v = 0; v <= 7; v++)
		histogram(0, v) = (float)histo[v] / current_features.cols;
	return histogram;
}
Mat_<float> get_features(Mat_<Vec3b> train_image)
{
	VlDsiftFilter* filter = vl_dsift_new_basic(train_image.rows, train_image.cols, step_size, nr_bins);

	Mat_<uchar> gray_image = rgb_to_gray(train_image);

	float* imgvec = gray_to_vector_normal(gray_image);

	vl_dsift_process(filter, imgvec);

	int descriptorSize = vl_dsift_get_descriptor_size(filter);
	int keypoint_nr = vl_dsift_get_keypoint_num(filter);

	float const* descriptors = vl_dsift_get_descriptors(filter);

	Mat_<float>img_descs(keypoint_nr, descriptorSize);

	for (int i = 0; i < keypoint_nr; i++)
	{
		for (int j = 0; j < descriptorSize; j++)
			img_descs(i, j) = descriptors[i*descriptorSize + j];
	}
	return img_descs;
}

std::vector<int> getBoundingBox(Mat_<Vec3b> im_src, int r, int f,float ratio, Mat_<float> h)
{

	Mat_<float> corner_pts(3, 4);
	corner_pts.setTo(1);
	corner_pts(1, 0) = 640 - (r + 1) * 80;
	corner_pts(0, 0) = f * 80;
	corner_pts(1, 1) = 640 - (r + 1) * 80;
	corner_pts(0, 1) = (f + 1) * 80;
	corner_pts(1, 2) = 640 - r * 80;
	corner_pts(0, 2) = (f + 1) * 80;
	corner_pts(1, 3) = 640 - r * 80;
	corner_pts(0, 3) = f * 80;	
	
	Mat_<float> pts = h.inv()*corner_pts;

	for (int j = 0; j < pts.cols; j++)
		for (int i = 0; i < pts.rows; i++)
			pts(i, j) /= pts(2, j);

	float width = pts(0, 2) - pts(0, 3);
	float sq_bottom = min(pts(1, 2), pts(1, 3));

	std::vector<int> bounding_box;
	int x1 = int(pts(0,3));
	int x2 = int(pts(0,2));
	int y1 = max(0, int(sq_bottom - width*ratio));
	int y2 = int(sq_bottom);

	bounding_box.push_back(x1);
	bounding_box.push_back(x2);
	bounding_box.push_back(y1);
	bounding_box.push_back(y2);
	return bounding_box;


}

void process_board(Mat_<float> points,Mat_<int> correct_board,int set_num,int num_pieces)
{
	Mat_<Vec3b> image = imread("images/" + std::to_string(num_pieces) + "_" + std::to_string(set_num) + ".jpg");
	cv::resize(image, image, cv::Size(image.cols * 0.2, image.rows * 0.2),0, 0);

	//imshow("image" + std::to_string(num_pieces) + "_" + std::to_string(set_num), image);

	Point2i imagePoint1(0.0f, 0.0f), imagePoint2(640.0f, 0.0f), imagePoint3(640.0f, 640.0f), imagePoint4(0.0f, 640.0f);

	std::vector<Point2i> imagePoints;
	imagePoints.push_back(imagePoint1);
	imagePoints.push_back(imagePoint2);
	imagePoints.push_back(imagePoint3);
	imagePoints.push_back(imagePoint4);


	std::vector<Point2i> boardPoints;
	Point2i firstPoint, secondPoint, thirdPoint, fourthPoint;
	firstPoint.x = points(0, 0);
	firstPoint.y = points(1, 0);

	secondPoint.x = points(0, 1);
	secondPoint.y = points(1, 1);

	thirdPoint.x = points(0, 2);
	thirdPoint.y = points(1, 2);

	fourthPoint.x = points(0, 3);
	fourthPoint.y = points(1, 3);

	boardPoints.push_back(firstPoint);
	boardPoints.push_back(secondPoint);
	boardPoints.push_back(thirdPoint);
	boardPoints.push_back(fourthPoint);

	Mat_<float> h = findHomography(boardPoints, imagePoints);

	Mat_<Vec3b> im_out(640,640);
	// Warp source image to destination based on homography
	warpPerspective(image, im_out, h, im_out.size());

	imshow("test",im_out);

	float ratio =aspect_ratios[4];

	int correct_pieces = 0;
	
	Mat_<int> board(8, 8);
	for(int r=0;r<=7;r++)
		for (int f=0; f <= 7; f++)
		{

			int x1, x2, y1, y2;
			
			/*
			std::vector<int> bounding = getBoundingBox(im_out, r, f, ratio, h);

			y2 = bounding.back();
			bounding.pop_back();
			y1 = bounding.back();
			bounding.pop_back();
			x2 = bounding.back();
			bounding.pop_back();
			x1 = bounding.back();
			bounding.pop_back();
			*/
			
			x1 = 640-80 * (f+2);
			x2 = 159;
			y1 = 80*r;
			y2 = 79;
			if (x1 < 0)
			{
				x1 = 0;
				x2 = 79;
			}
			//std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << "\n";
			Rect crop;
			crop.x = y1;
			crop.y = x1;
			crop.width = y2;
			crop.height = x2;

			Mat_<Vec3b> cropped = im_out(crop);

			Mat_<Vec3b> resized(64, 128);
			cv::resize(cropped, resized, cv::Size(64, 128), 0, 0);

			//imshow("test_crop", resized);
			//waitKey(0);
			Mat_<float> features = get_features(resized);

			Mat_<float> represent;
			represent=get_histo(features);


			float maxDist = -1.0f;
			int minLabel = -1;

			Mat res;   // output
			
			for (int p = 0; p < 7; p++)
			{
				classifiers[p]->predict(represent, res, cv::ml::StatModel::RAW_OUTPUT);

				if (res.at<float>(0, 0) > maxDist )
				{
					maxDist = res.at<float>(0, 0);
					minLabel = p;
				}

				/*float response=svm_all->predict(represent);
				minLabel = response;*/
			}

			board(r, f) = minLabel;

			if (minLabel == correct_board(r, f))
				correct_pieces++;
		}


	std::cout << board << "\n";

	std::cout << "Correctly predicted pieces: "<<correct_pieces << "/64\n\n\n";
	waitKey(0);



}

Mat_<float> transform_to_Mat(std::vector<Mat> features,int elems_in_each,int cols)
{
	Mat_<float> final_features(features.size()*elems_in_each, cols);

	int current_index = 0;
	for (Mat_<float> f : features)
		final_features.push_back(f);
	return final_features;
}


Ptr<ml::SVM> get_svm(Mat_<float> training_data,Mat_<int> labels,Mat_<float> weights)
{
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::POLY);
	svm->setGamma(3);
	svm->setDegree(2);
	svm->setClassWeights(weights);

	svm->train(training_data, ml::ROW_SAMPLE, labels);

	return svm;
}

/*
Ptr<ml::SVM> get_all_svm(Mat_<float> training_data, Mat_<int> labels, Mat_<float> weights)
{
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setClassWeights(weights);

	svm->train(training_data, ml::ROW_SAMPLE, labels);

	return svm;
}*/

void training()
{
	std::vector<Mat> features;


	for (String piece : pieces)
	{
		std::vector<cv::String> fn;

		glob("images/training_images/" + piece + "/*.jpg", fn, false);

		size_t count = fn.size();
		for (size_t i = 0; i < count; i++)
		{

			Mat_<Vec3b> train_image = imread(fn[i]);

			cv::resize(train_image, train_image, cv::Size(64, 128), 0, 0);
			//imshow("altceva", train_image);
			//waitKey(0);


			Mat_<float>img_descs = get_features(train_image);

			features.push_back(img_descs);
		}
	}

	Mat_<float> clustering_features = transform_to_Mat(features,features.front().rows, features.front().cols);


	cv::Mat labels;

	int k = 8; // searched clusters in k-means

	cv::kmeans(clustering_features, k, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 0.001), 2, cv::KMEANS_PP_CENTERS, centers);
	
	int samples = features.size();
	int img_nr = 0;
	Mat_<float> histograms(samples,8);
	Mat_<int> lbls(samples, 1);

	for (int p=0;p<7;p++)
	{
		String piece = pieces[p];
		std::vector<cv::String> fn;

		glob("images/training_images/" + piece + "/*.jpg", fn, false);

		size_t count = fn.size();
		for (size_t i = 0; i < count; i++)
		{
			int histo[8] = { 0,0,0,0,0,0,0,0 };
			

			Mat_<float> current_features = features[img_nr];
			for (int j = 0; j < current_features.rows;j++)
			{
				Mat_<float> feature_vector(1,128);
				for (int q = 0; q < current_features.cols; q++)
					feature_vector(0, q) = current_features(j, q);

				histo[find_closest_center(feature_vector, centers)]++;

			}
			for(int v=0;v<=7;v++)
				histograms(img_nr,v) = (float)histo[v]/current_features.cols ;
			lbls(img_nr,0) = p;
			img_nr++;
		}
	}
	
	std::cout << lbls << "\n";
	Mat_<float> weights(1, 7);
	for (int p = 0; p<7; p++)
		weights(0,p)= piece_weights[2 * p+1];

	for (int p = 0; p < 7; p++)
	{
		Mat_<int> piece_label(lbls.rows,1);

		for (int q = 0; q < lbls.rows; q++)
			if (lbls(q, 0) != p)
				piece_label(q, 0) = -1;
			else
				piece_label(q, 0) = 1;

		Mat_<float> weights(1, 2);
		weights(0, 0) = piece_weights[2 * p];
		weights(0, 1) = piece_weights[2 * p+1];
		classifiers[p] = get_svm(histograms,piece_label,weights);
	}


}

void proiect_start()
{
	for (int i = 1; i <= 5; i++)
	{
		process_board(matPts_5_1, matBoard_5_1, i, 5);
		process_board(matPts_10_1, matBoard_10_1, i, 10);
		process_board(matPts_15_1, matBoard_15_1, i, 15);
		process_board(matPts_20_1, matBoard_20_1, i, 20);
		process_board(matPts_25_1, matBoard_25_1, i, 25);
		process_board(matPts_30_1, matBoard_30_1, i, 30);
	}
	getchar();
	getchar(); 
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Run project\n");
		printf(" 2 - Train\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				proiect_start();
				break;
			case 2:
				training();
				break;
		}
	}
	while (op!=0);
	return 0;
}