#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"
#include "person_detector.hpp"

PersonDetector::PersonDetector(const std::string& path) {
  this->path = path;
  this->background_image = PersonDetector::load_image(0, "depth", path);
}

void PersonDetector::detect_people(int image_id, cv::Mat& output, unsigned int& best_K) {
  cv::Mat depth_image = PersonDetector::load_image(image_id, "depth", this->path);

  // remove background
  depth_image = this->background_image - depth_image;

  // rescale image to [0, 1]
  double min, max;
  cv::minMaxLoc(depth_image, &min, &max);
  depth_image = (depth_image - min) / (max - min);

  // remove border noise: detect high derivative points of different
  // sign and close to each other with second order derivative
  cv::Mat first_derivative, second_derivative;
  cv::Sobel(depth_image, first_derivative, CV_32F, 1, 0);
  cv::Laplacian(depth_image, second_derivative, CV_32F, 3);

  cv::absdiff(first_derivative, cv::Scalar::all(0), first_derivative);
  cv::absdiff(second_derivative, cv::Scalar::all(0), second_derivative);

  cv::threshold(first_derivative, first_derivative, 0.75, 1., cv::THRESH_BINARY);
  cv::threshold(second_derivative, second_derivative, 0.25, 1., cv::THRESH_BINARY);

  cv::Mat mask = (first_derivative > 0) & (second_derivative > 0);

  cv::dilate(mask, mask, cv::Mat());
  cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY_INV);

  // duplicate depth_image, but with its mean color
  cv::Mat temp(depth_image.rows, depth_image.cols,
              depth_image.type(), cv::mean(depth_image));

  // draw everythin but the noise, filled by mean image value
  depth_image.copyTo(temp, mask);
  depth_image = temp;

  std::vector<cv::Point2f> points;

  // compute threshold based on maximum value
  cv::minMaxLoc(depth_image, &min, &max);
  float threshold = max * 0.6;

  for (int row = 0; row < depth_image.rows; row++) {
    for (int col = 0; col < depth_image.cols; col++) {
      float value = depth_image.at<float>(row, col);

      if (value > threshold) {
        points.push_back(cv::Point2f((float) row, (float) col));
      }
    }
  }

  // evaluate Kmeans clustering for different values of K
  std::vector<unsigned int> Ks = {1, 2, 3, 4, 5};

  float best_score = std::numeric_limits<float>::infinity();
  std::vector<int> best_labels;

  cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);

  double score, compactness;
  double eta = 0.42;
  for (auto& K: Ks) {
    std::vector<int> labels;
    compactness = cv::kmeans(points, K, labels, criteria, 10, cv::KMEANS_PP_CENTERS);
    score = std::log(compactness) + K * eta;

    if (score < best_score) {
      best_K = K;
      best_score = score;
      best_labels = labels;
    }
  }

  // paint color image with found clustering
  output = PersonDetector::load_image(image_id, "color", this->path);

  std::vector<cv::Vec3b> cluster_colors = { cv::Vec3b(39, 39, 216),
                                          cv::Vec3b(39, 201, 216),
                                          cv::Vec3b(39, 216, 115),
                                          cv::Vec3b(168, 216, 39),
                                          cv::Vec3b(216, 130, 39),
                                          cv::Vec3b(216, 39, 71),
                                          cv::Vec3b(139, 39, 216) };

  for (unsigned int i = 0; i < points.size(); i++) {
    int col = (int) points[i].y;
    int row = (int) points[i].x;

    output.at<cv::Vec3b>(row, col) = 0.5 * output.at<cv::Vec3b>(row, col) +
                                    0.5 * cluster_colors[best_labels[i]];
  }
}

// std::cout << points[0] << std::endl;
