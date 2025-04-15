//
// Created by bowserj on 4/15/25.
//

#ifndef BASEWEIGHTSNAP_IMAGEPROCESSOR_H
#define BASEWEIGHTSNAP_IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>


class ImageProcessor {
public:
    static cv::Mat loadImage(const std::string& path);
    static std::vector<float> preprocess(const cv::Mat& image, int target_width, int target_height);

};


#endif //BASEWEIGHTSNAP_IMAGEPROCESSOR_H
