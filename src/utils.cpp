#include "utils.hpp"

void my_utils::show(const std::string& name, cv::Mat& img,
                   bool store,
                   float max_width,
                   float max_height) {
    cv::Mat resized_img;

    float width_ratio = img.size().width / max_width;
    float height_ratio = img.size().height / max_height;

    float max_ratio = std::max(width_ratio, height_ratio);
    float factor = 1 / max_ratio;

    cv::resize(img, resized_img, cv::Size(), factor, factor);

    cv::namedWindow(name);
    cv::imshow(name, resized_img);

    if (store) {
        cv::imwrite("../results/" + name + ".png", img);
    }
}

void my_utils::wait_for(char progress_key) {
    std::cout << "Press <" << progress_key
             << "> to proceed" << std::endl;
    char command;
    do {
        command = (char) cv::waitKey();
        std::cout << "<" << command << "> key pressed" << std::endl;
    } while (command != progress_key);
}
