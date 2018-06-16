#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace my_utils {
    /**
     * Show resized image, to fit page
     *
     * @param name   wanted window name
     * @param img    BGR image
     * @param factor resizing factor for image to be displayed
     * @param store  wheater or not save the image to default path
     */
    void show(const std::string& name, cv::Mat& img,
              bool store = true,
              float max_width = 1000.0,
              float max_height = 700.0);

    /**
     * Wait for user input of a specific key
     *
     * @param progress_key  key to proceed after function call
     */
    void wait_for(char progress_key);
};

#endif
