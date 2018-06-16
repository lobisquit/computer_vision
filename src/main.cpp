
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"
#include "person_detector.hpp"

using namespace my_utils;

enum WorkingMode { SHOW_ALL, SHOW_ERRORS, SAVE };

int main(int argc, char** argv) {
  // get correct working mode, defaulting to SHOW_ALL
  WorkingMode wm;
  if (argc <= 1 || std::string("show-all").compare(argv[1]) == 0) {
    wm = WorkingMode::SHOW_ALL;
  }
  else if (std::string("show-errors").compare(argv[1]) == 0) {
    wm = WorkingMode::SHOW_ERRORS;
  }
  else if (std::string("save").compare(argv[1]) == 0) {
    wm = WorkingMode::SAVE;
  }
  else {
    wm = WorkingMode::SHOW_ALL;
  }

  std::string image_path = "../dataset/";
  PersonDetector detector = PersonDetector(image_path);

  // ground truth
  std::vector<unsigned int> true_Ks = {0, 1, 3, 3,
                                      2, 2, 2, 3,
                                      4, 2, 4, 3,
                                      4, 3, 2, 2,
                                      3, 3};

  unsigned int correct = 0;
  unsigned int total = 0;
  for (int image_id = 1; image_id <= 17; image_id++) {
    cv::Mat out;
    unsigned int best_K;
    detector.detect_people(image_id, out, best_K);

    bool error = best_K != true_Ks[image_id];
    if (error) {
      std::cout << "In image " << std::to_string(image_id) << ", "
               << std::to_string(best_K) << " people were found but "
               << true_Ks[image_id] << " were expected"
               << std::endl;
    }
    else {
      correct += 1;
    }
    total += 1;

    // report, as specified in WorkingMode
    std::string window_name = "detection-" + std::to_string(image_id);
    if (wm == WorkingMode::SHOW_ALL || (wm == WorkingMode::SHOW_ERRORS && error)) {
      my_utils::show(window_name, out, false);
      my_utils::wait_for('o');
      cv::destroyWindow(window_name);
    }
    else if (wm == WorkingMode::SAVE) {
      my_utils::show(window_name, out, true);
    }
  }
  std::cout << std::to_string(correct) << "/" << std::to_string(total)
           << " images are correctly processed" << std::endl;
}
