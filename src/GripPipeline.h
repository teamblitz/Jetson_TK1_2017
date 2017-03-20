#pragma once
//#include "vision/VisionRunner.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <string>
#include <math.h>

namespace grip {

/**
* GripPipeline class.
* 
* An OpenCV pipeline generated by GRIP.
*/
class GripPipeline { //: public frc::VisionPipeline {
	private:
		cv::Mat cvResizeOutput;
		cv::Mat hslThresholdOutput;
		cv::Mat cvDilateOutput;
		std::vector<std::vector<cv::Point> > findContoursOutput;
		std::vector<std::vector<cv::Point> > filterContoursOutput;
		void cvResize(cv::Mat &, cv::Size &, double , double , int , cv::Mat &);
		void hslThreshold(cv::Mat &, double [], double [], double [], cv::Mat &);
		void cvDilate(cv::Mat &, cv::Mat &, cv::Point &, double , int , cv::Scalar &, cv::Mat &);
		void findContours(cv::Mat &, bool , std::vector<std::vector<cv::Point> > &);
		void filterContours(std::vector<std::vector<cv::Point> > &, double , double , double , double , double , double , double [], double , double , double , double , std::vector<std::vector<cv::Point> > &);

	public:
		GripPipeline();
		void Process(cv::Mat& source0); // override;
		cv::Mat* GetCvResizeOutput();
		cv::Mat* GetHslThresholdOutput();
		cv::Mat* GetCvDilateOutput();
		std::vector<std::vector<cv::Point> >* GetFindContoursOutput();
		std::vector<std::vector<cv::Point> >* GetFilterContoursOutput();
};


} // end namespace grip


