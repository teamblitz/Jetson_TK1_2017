#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

#define USE_CAMERA_INPUT 0
#define DEBUG_MODE 0

// Forward declarations.
cv::SimpleBlobDetector::Params getParamsForGRIPFindBlobs();
cv::SimpleBlobDetector::Params getParamsForNormalVideo();
cv::SimpleBlobDetector::Params getParamsForThresholdVideo();

int main()
{
	// Open USB camera on port 0.
#if USE_CAMERA_INPUT
    cv::VideoCapture input(0);
#else
//	cv::VideoCapture input("../sample_media/videos/WIN_20170307_20_43_09_Pro.mp4");
    cv::VideoCapture input("../sample_media/videos/WIN_20170307_20_45_18_Pro.mp4");
#endif

	// Grab and process frames.
	for (;;)
	{
		cv::Mat frame;
		if (!input.read(frame))
			break;

		cv::Mat blurredFrame;
		cv::medianBlur(frame, blurredFrame, 11);
	
		cv::Mat thresholdFrame;
		cv::threshold(blurredFrame, thresholdFrame, 220, 255, CV_THRESH_BINARY);

		std::vector<cv::KeyPoint> keypoints;

		cv::SimpleBlobDetector::Params params = getParamsForGRIPFindBlobs();
		
		cv::SimpleBlobDetector detector(params);
		detector.detect(thresholdFrame, keypoints); 

		// Test code to display points on frame.
		std::vector<cv::Point2f> points;
		cv::KeyPoint::convert(keypoints, points);

		cv::Mat detectionFrame;
		frame.copyTo(detectionFrame);
		//std::cout << "Keypoints " << keypoints.size() << "\n";
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			cv::circle(detectionFrame, keypoints[i].pt, keypoints[i].size/2, 128, 3);
		}

#if DEBUG_MODE
		cv::imshow("frame", detectionFrame);
		cv::waitKey(1);
#endif
	}
}

cv::SimpleBlobDetector::Params getParamsForGRIPFindBlobs()
{
    cv::SimpleBlobDetector::Params params = cv::SimpleBlobDetector::Params();
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

cv::SimpleBlobDetector::Params getParamsForNormalVideo()
{
	cv::SimpleBlobDetector::Params params = cv::SimpleBlobDetector::Params();
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

cv::SimpleBlobDetector::Params getParamsForThresholdVideo()
{
    cv::SimpleBlobDetector::Params params = cv::SimpleBlobDetector::Params();
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
