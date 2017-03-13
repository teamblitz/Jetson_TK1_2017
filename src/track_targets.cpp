#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define USE_CAMERA_INPUT 0
#define DEBUG_MODE 1

// Forward declarations.
vector<KeyPoint> filterKeyPoints(vector<KeyPoint> keypoints);
SimpleBlobDetector::Params getParamsForGRIPFindBlobs();
SimpleBlobDetector::Params getParamsForNormalVideo();
SimpleBlobDetector::Params getParamsForThresholdVideo();

int main(int argc, char** argv)
{
	cout << argv[0] << " running..." << endl;

#if USE_CAMERA_INPUT
    // Open USB camera on port 0.
    VideoCapture input(0);
	if (!input.isOpened())
	{
		cerr << "ERROR: Failed to open camera!" << endl;
		cout << "Make sure that there are no other instances of this program already running!" << endl;
		return -1;
	}
#else
	// Open a test video file.
//	VideoCapture input("../sample_media/videos/WIN_20170307_20_43_09_Pro.mp4");
    VideoCapture input("../sample_media/videos/WIN_20170307_20_45_18_Pro.mp4");
	if (!input.isOpened())
	{
		cout << "Could not open test video file. Reverting to live camera feed." << endl;
		input.open(0);
		if (!input.isOpened())
		{
        	cerr << "ERROR: Failed to open camera!" << endl;
        	cout << "Make sure that there are no other instances of this program already running!" << endl;
        	return -1;
		}
	}
#endif

	// Grab and process frames.
	for (;;)
	{
		Mat frame;
		if (!input.read(frame))
			break;

		Mat blurredFrame;
		medianBlur(frame, blurredFrame, 11);
	
		Mat thresholdFrame;
		threshold(blurredFrame, thresholdFrame, 220, 255, CV_THRESH_BINARY);

		SimpleBlobDetector::Params params = getParamsForGRIPFindBlobs();
		SimpleBlobDetector detector(params);

		vector<KeyPoint> keypoints;
		detector.detect(thresholdFrame, keypoints); 

		keypoints = filterKeyPoints(keypoints);

#if DEBUG_MODE
		// In debug mode, render the keypoints onto the frames.
		Mat detectionFrame;
		frame.copyTo(detectionFrame);
		//cout << "Keypoints " << keypoints.size() << endl;
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			circle(detectionFrame, keypoints[i].pt, keypoints[i].size/2, 128, 3);
		}

		imshow("frame", detectionFrame);
		waitKey(1);
#endif
	}

	// Clean up and shutdown.
	input.release();
	cout << argv[0] << " finished!" << endl;
}

vector<KeyPoint> filterKeyPoints(vector<KeyPoint> keypoints)
{
	vector<KeyPoint> newKeypoints;

	if (keypoints.size() <= 1)
	{
		return newKeypoints;
	}

	sort(keypoints.begin(), keypoints.end(), 
		[] (KeyPoint const& a, KeyPoint const& b) { return a.size > b.size; });

	for (auto iter = keypoints.begin(); iter != keypoints.end()-1; ++iter)
	{
		for (auto other = iter+1; other != keypoints.end(); ++other)
		{
			auto avgTargetSize = ((*iter).size + (*other).size)/2;
			auto deltaX = abs((*iter).pt.x - (*other).pt.x);
			auto deltaY =  abs((*iter).pt.y - (*other).pt.y);
			if ((deltaX < 7 * avgTargetSize) && (deltaX > 2 * avgTargetSize) &&
				(deltaY < 3 * avgTargetSize))
			{
				newKeypoints.push_back(*iter);
				newKeypoints.push_back(*other);
				return newKeypoints;
			}
		}	
	}

	return newKeypoints;
}

SimpleBlobDetector::Params getParamsForGRIPFindBlobs()
{
    SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
    params.thresholdStep = 10;				// 10
    params.minThreshold = 50;				// 50
    params.maxThreshold = 220;				// 220
    params.minRepeatability = 2;			// 2
    params.minDistBetweenBlobs = 10;		// 10
    params.filterByColor = true;			// true
    params.blobColor = 255;					// 255
    params.filterByArea = true;				// true
    params.minArea = 1000;					// 1000
	params.maxArea = 20000;					// INT_MAX
    params.filterByCircularity = true;		// true
	params.minCircularity = 0;				// 0
	params.maxCircularity = 1;				// 1
    params.filterByInertia = true;			// true
	params.minInertiaRatio = 0.1;			// 0.1
	params.maxInertiaRatio = INT_MAX;		// INT_MAX
    params.filterByConvexity = true;		// true
    params.minConvexity = 0.95;				// 0.95
    params.maxConvexity = INT_MAX;			// INT_MAX

    return params;
}

SimpleBlobDetector::Params getParamsForNormalVideo()
{
	SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
    params.thresholdStep = 2;
    params.minThreshold = 180;
    params.maxThreshold = 255;
    params.minRepeatability = 1;
    params.minDistBetweenBlobs = 20;
    params.filterByColor = false;
    params.blobColor = 100;
    params.filterByArea = true;
    params.minArea = 400;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.maxConvexity = 1.0;

	return params;
}

SimpleBlobDetector::Params getParamsForThresholdVideo()
{
    SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
    params.thresholdStep = 127;
    params.minThreshold = 0;
    params.maxThreshold = 255;
    params.minRepeatability = 1;
    params.minDistBetweenBlobs = 20;
    params.filterByColor = false;
    params.blobColor = 100;
    params.filterByArea = true;
    params.minArea = 400;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.maxConvexity = 1.0;

    return params;
}
