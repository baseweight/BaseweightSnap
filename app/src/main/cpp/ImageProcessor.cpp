//
// Created by bowserj on 4/15/25.
//

#include "ImageProcessor.h"
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat ImageProcessor::loadImage(const std::string& path) {
    return cv::imread(path);
}

std::vector<float> ImageProcessor::preprocess(const cv::Mat& image, int target_width = 224, int target_height = 224) {
    // Resize and normalize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width, target_height));

    // Convert to float and normalize
    cv::Mat floatImage;
    resized.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Normalize with mean and std
    // Values were taken from config loaded from here:
    // https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/discussions/14
    std::vector<float> mean = {0.5, 0.5, 0.5};
    std::vector<float> std = {0.5, 0.5, 0.5};

    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }

    cv::merge(channels, floatImage);

    // Convert to NCHW format expected by ONNX
    std::vector<float> input_tensor(1 * 3 * target_height * target_width);

    // Copy data to input tensor (HWC to CHW)
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                input_tensor[c * target_height * target_width + h * target_width + w] =
                        floatImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return input_tensor;
}