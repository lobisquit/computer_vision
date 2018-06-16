#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

class PersonDetector {
private:
  cv::Mat background_image;
  cv::Mat color_image;
  std::string path;

  static cv::Mat load_image(int image_id, const std::string& kind, const std::string& path) {
    std::string image_path = path + kind + std::to_string(image_id) + ".png";

    cv::Mat image;
    if (kind.compare("depth") == 0) {
      // convert image to float, scaling values to [0, 1]
      cv::Mat original_image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
      original_image.convertTo(image, CV_32FC1, 1/65355.0);
    }
    if (kind.compare("color") == 0) {
      image = cv::imread(image_path, cv::IMREAD_COLOR);
    }

    return image;
  };

public:
  PersonDetector(const std::string& path);

  void detect_people(int image_id, cv::Mat& output, unsigned int& K);
};

#endif
