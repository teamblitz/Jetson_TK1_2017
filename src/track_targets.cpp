#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Code conditionals.
#define USE_CAMERA_INPUT 0
#define VIEW_OUTPUT 1
#define MEASURE_PERFORMANCE 1

// Performance macros.
#if MEASURE_PERFORMANCE
#define TICK_ACCUMULATOR_START(NAME)    auto NAME ## Start = getTickCount()
#define TICK_ACCUMULATOR_END(NAME)      NAME ## Ticks += (getTickCount() - NAME ## Start)
#else
#define TICK_ACCUMULATOR_START(NAME)
#define TICK_ACCUMULATOR_END(NAME)
#endif

// Constants.
static const float FRAME_SCALE_FACTOR = 0.5;

// Forward declarations.
void filterKeyPoints(vector<KeyPoint> const keypoints, vector<KeyPoint>& hits, vector<KeyPoint>& skips);
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
//  VideoCapture input("../sample_media/videos/WIN_20170307_20_43_09_Pro.mp4");
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

    SimpleBlobDetector::Params params = getParamsForGRIPFindBlobs();
    SimpleBlobDetector detector(params);

#if MEASURE_PERFORMANCE
    auto start = getTickCount();
    int64 readTicks = 0;
    int64 resizeTicks = 0;
    int64 blurTicks = 0;
    int64 thresholdTicks = 0;
    int64 detectTicks = 0;
    int64 sortTicks = 0;
    int64 filterTicks = 0;
    int64 viewTicks = 0;
    int64 frameCount = 0;
#endif

    // Grab and process frames.
    for (;;)
    {
        TICK_ACCUMULATOR_START(read);
        Mat rawFrame;
        if (!input.read(rawFrame))
            break;
        TICK_ACCUMULATOR_END(read);
#if MEASURE_PERFORMANCE
        frameCount++;
#endif

        TICK_ACCUMULATOR_START(resize);
        Mat frame;
        resize(rawFrame, frame, Size(), FRAME_SCALE_FACTOR, FRAME_SCALE_FACTOR, CV_INTER_AREA);
        TICK_ACCUMULATOR_END(resize);

        TICK_ACCUMULATOR_START(blur);
        //Mat blurredFrame;
        //medianBlur(frame, blurredFrame, 11);
        TICK_ACCUMULATOR_END(blur);
    
        TICK_ACCUMULATOR_START(threshold);
        //Mat thresholdFrame;
        //threshold(frame, thresholdFrame, 200, 255, CV_THRESH_BINARY);
        TICK_ACCUMULATOR_END(threshold);

        TICK_ACCUMULATOR_START(detect);
        vector<KeyPoint> keyPoints;
        detector.detect(frame, keyPoints); 
        TICK_ACCUMULATOR_END(detect);

        TICK_ACCUMULATOR_START(sort);
        sort(keyPoints.begin(), keyPoints.end(),
             [] (KeyPoint const& a, KeyPoint const& b) { return a.size > b.size; });
        TICK_ACCUMULATOR_END(sort);

        TICK_ACCUMULATOR_START(filter);
        vector<KeyPoint> hits, misses;
        filterKeyPoints(keyPoints, hits, misses);
        TICK_ACCUMULATOR_END(filter);

#if VIEW_OUTPUT
        // In debug mode, render the keypoints onto the frames.
        TICK_ACCUMULATOR_START(view);
        Mat detectionFrame;
        frame.copyTo(detectionFrame);
        //cout << "Keypoints " << keypoints.size() << endl;
        for (size_t i = 0; i < hits.size(); i++)
        {
            circle(detectionFrame, hits[i].pt, hits[i].size/2, Scalar(0, 0, 255), 3);
        }

        for (size_t i = 0; i < misses.size(); i++)
        {
            circle(detectionFrame, misses[i].pt, misses[i].size/2, Scalar(0, 0, 0), 3);
        }

        imshow("frame", detectionFrame);
        waitKey(1);
        TICK_ACCUMULATOR_END(view);
#endif
    }

#if MEASURE_PERFORMANCE
    auto final = getTickCount();
    auto const tickFreq = getTickFrequency();
    auto totalTime = (final - start)/getTickFrequency();    // seconds
    auto other = final - start;
    cout << "Execution took " << totalTime << " seconds" << endl;
    cout << "  Read: " << readTicks/tickFreq << " seconds" << endl;
    other -= readTicks;
    cout << "  Resize: " << resizeTicks/tickFreq << " seconds" << endl;
    other -= resizeTicks;
    cout << "  Blur: " << blurTicks/tickFreq << " seconds" << endl;
    other -= blurTicks;
    cout << "  Threshold: " << thresholdTicks/tickFreq << " seconds" << endl;
    other -= thresholdTicks;
    cout << "  Sort: " << sortTicks/tickFreq << " seconds" << endl;
    other -= sortTicks;
    cout << "  Detect: " << detectTicks/tickFreq << " seconds" << endl;
    other -= detectTicks;
    cout << "  Filter: " << filterTicks/tickFreq << " seconds" << endl;
    other -= filterTicks;
#if VIEW_MOD
    cout << "  View: " << viewTicks/tickFreq << " seconds" << endl;
    other -= viewTicks;
#endif
    cout << "  Other: " << other/tickFreq << " seconds" << endl;
    cout << "Frames processed: " << frameCount << endl;
    cout << "Frame rate: " << frameCount/totalTime << " frames/second" << endl;
#endif

    // Clean up and shutdown.
    input.release();
    cout << argv[0] << " finished!" << endl;
}

/**
 * Filter the keypoints looking for potential keypoints that
 * correspond to the two matching targets. Once found, the
 * target keypoints will be returns in the "hits" vector.i
 * Any keypoints skipped up until that point (because they
 * are not viable target candidates) will be returned in the
 * "skips" vector.
 */
void filterKeyPoints(vector<KeyPoint> const keyPoints, vector<KeyPoint>& hits, vector<KeyPoint>& skips)
{
    if (keyPoints.size() == 0)
    {
        return;
    }

    if (keyPoints.size() == 1)
    {
        skips.push_back(*(keyPoints.begin()));
        return;
    }

    for (auto iter = keyPoints.begin(); iter != keyPoints.end()-1; iter++)
    {
        for (auto other = iter+1; other != keyPoints.end(); other++)
        {
            auto avgTargetSize = ((*iter).size + (*other).size)/2;
            auto deltaX = abs((*iter).pt.x - (*other).pt.x);
            auto deltaY =  abs((*iter).pt.y - (*other).pt.y);
            if ((deltaX < 9 * avgTargetSize) && (deltaX > 2 * avgTargetSize) &&
                (deltaY < 3 * avgTargetSize))
            {
                hits.push_back(*iter);
                hits.push_back(*other);
                return;
            }
            else
            {
                skips.push_back(*iter);
            }
        }   
    }
}

SimpleBlobDetector::Params getParamsForGRIPFindBlobs()
{
    const float fsfs = FRAME_SCALE_FACTOR * FRAME_SCALE_FACTOR;

    SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
    params.thresholdStep = 10;              // 10
    params.minThreshold = 50;               // 50
    params.maxThreshold = 220;              // 220
    params.minRepeatability = 2;            // 2
    params.minDistBetweenBlobs = 10;        // 10
    params.filterByColor = true;            // true
    params.blobColor = 255;                 // 255
    params.filterByArea = true;             // true
    params.minArea = 1000 * fsfs;           // 1000
    params.maxArea = 20000 * fsfs;          // INT_MAX
    params.filterByCircularity = true;      // true
    params.minCircularity = 0;              // 0
    params.maxCircularity = 1;              // 1
    params.filterByInertia = true;          // true
    params.minInertiaRatio = 0.1;           // 0.1
    params.maxInertiaRatio = INT_MAX;       // INT_MAX
    params.filterByConvexity = true;        // true
    params.minConvexity = 0.95;             // 0.95
    params.maxConvexity = INT_MAX;          // INT_MAX

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
