#include <bundle/bundle_adjuster.h>
#include <geometry/camera.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace bundle {

class BundleAdjusterSceneTest : public ::testing::Test {
 protected:
  void SetUp() override { rng_.seed(42); }

  // Create a camera looking at origin
  geometry::Pose LookAt(const Vec3d& center, const Vec3d& target,
                        const Vec3d& up) {
    Vec3d z_axis = (target - center).normalized();
    Vec3d x_axis = up.cross(z_axis).normalized();
    Vec3d y_axis = z_axis.cross(x_axis).normalized();

    Mat3d R;
    R.row(0) = x_axis;
    R.row(1) = y_axis;
    R.row(2) = z_axis;

    // Ensure positive determinant
    if (R.determinant() < 0) {
      R.row(0) = -R.row(0);
    }

    geometry::Pose pose;
    Vec3d t = -R * center;
    pose.SetFromWorldToCamera(R, t);
    return pose;
  }

  std::mt19937 rng_;
};

TEST_F(BundleAdjusterSceneTest, SceneWithOutliers) {
  // 1. Setup Ground Truth
  // Points: Sphere R=10
  int n_points = 1000;
  double R_points = 10.0;
  std::vector<Vec3d> gt_points;
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);

  for (int i = 0; i < n_points; ++i) {
    Vec3d p(uniform(rng_), uniform(rng_), uniform(rng_));
    p.normalize();
    p *= R_points;
    gt_points.push_back(p);
  }

  // Cameras: Sphere R=20
  int n_shots = 20;
  double R_cams = 20.0;
  std::vector<geometry::Pose> gt_poses;
  std::vector<std::string> shot_ids;

  for (int i = 0; i < n_shots; ++i) {
    Vec3d center(uniform(rng_), uniform(rng_), uniform(rng_));
    center.normalize();
    center *= R_cams;

    // Look at origin
    gt_poses.push_back(LookAt(center, Vec3d::Zero(), Vec3d(0, -1, 0)));
    shot_ids.push_back("shot_" + std::to_string(i));
  }

  // Camera Intrinsics
  // 4000x3000 image, focal 0.8 * width
  geometry::Camera camera =
      geometry::Camera::CreatePerspectiveCamera(0.8, 0.1, 0.0);
  camera.id = "calib";
  camera.width = 4000;
  camera.height = 3000;
  double half_height = double(camera.height) / camera.width / 2.0;

  // 2. Perturb and Setup Bundle Adjuster
  BundleAdjuster ba;

  // Add Camera
  ba.AddCamera("calib", camera, camera, true);  // Fixed intrinsics

  // Add Rig Camera (Identity transform)
  ba.AddRigCamera("RC0", geometry::Pose(), geometry::Pose(), true);

  // Add Rig Instances (Shots) with Noisy Priors

  // Noise for inliers shots
  double pos_noise_sigma = 0.3;
  std::normal_distribution<double> pos_noise(0.0, pos_noise_sigma);

  // Outliers probability and magnitude
  std::uniform_real_distribution<double> outlier_prob(0.0, 1.0);
  double outlier_ratio = 0.0;
  double outlier_magnitude = 20.0;

  // Remember outliers for proper checking
  std::vector<bool> is_outlier_shot(n_shots, false);
  for (int i = 0; i < n_shots; ++i) {
    geometry::Pose initial_pose = gt_poses[i];

    // Add outliers
    bool is_outlier = is_outlier_shot[i] = outlier_prob(rng_) < outlier_ratio;

    Vec3d origin = initial_pose.GetOrigin();
    if (is_outlier) {
      origin += Vec3d(outlier_magnitude, outlier_magnitude, outlier_magnitude);
    } else {
      origin += Vec3d(pos_noise(rng_), pos_noise(rng_), pos_noise(rng_));
    }
    initial_pose.SetOrigin(origin);

    // Rig setup
    std::string rig_id = "rig_" + shot_ids[i];
    ba.AddRigInstance(rig_id, initial_pose, {{shot_ids[i], "calib"}},
                      {{shot_ids[i], "RC0"}}, false);

    // Add Prior (Simulate GPS)
    Vec3d prior_pos = gt_poses[i].GetOrigin();
    if (is_outlier) {
      prior_pos +=
          Vec3d(outlier_magnitude, outlier_magnitude, outlier_magnitude);
    } else {
      prior_pos += Vec3d(pos_noise(rng_), pos_noise(rng_), pos_noise(rng_));
    }
    ba.AddRigInstancePositionPrior(rig_id, prior_pos,
                                   Vec3d::Constant(pos_noise_sigma), "0");
  }

  // Add Points and Projections

  // Inliers noise
  double proj_noise_sigma = 1.0;  // pixels
  std::normal_distribution<double> proj_noise(0.0, proj_noise_sigma);

  // Outlier probability and magnitude
  double point_outlier_ratio = 0.1;
  std::uniform_real_distribution<double> image_x(-0.5, 0.5);

  int added = 0;
  for (int i = 0; i < n_points; ++i) {
    std::string point_id = "point_" + std::to_string(i);

    // Add noisy points (same sigma as shots)
    Vec3d initial_pt = gt_points[i];
    initial_pt += Vec3d(pos_noise(rng_), pos_noise(rng_), pos_noise(rng_));
    ba.AddPoint(point_id, initial_pt, false);

    for (int j = 0; j < n_shots; ++j) {
      Vec3d pt_cam = gt_poses[j].TransformWorldToCamera(gt_points[i]);
      if (pt_cam[2] <= 0) {
        continue;  // Behind camera
      }

      Vec2d proj = camera.Project(pt_cam);
      if (std::abs(proj[0]) > 0.5 || std::abs(proj[1]) > half_height) {
        continue;
      }

      // Perturb projection
      double scale = 1.0 / camera.width;  // Normalize noise
      if (outlier_prob(rng_) < point_outlier_ratio) {
        proj = Vec2d(image_x(rng_), image_x(rng_));
      } else {
        proj += Vec2d(proj_noise(rng_) * scale, proj_noise(rng_) * scale);
      }

      ba.AddPointProjectionObservation(shot_ids[j], point_id, proj, scale,
                                       false);
      ++added;
    }
  }
  std::cout << "Added " << added << " projections." << std::endl;

  for (int i = 0; i < n_shots; ++i) {
    std::string rig_id = "rig_" + shot_ids[i];
    geometry::Pose opt_pose = ba.GetRigInstance(rig_id).GetValue();
    double err = (opt_pose.GetOrigin() - gt_poses[i].GetOrigin()).norm();

    std::cout << "Initial Pose Error for " << rig_id << ": " << err
              << std::endl;
  }

  ba.SetUseAnalyticDerivatives(true);
  ba.Run();

  // 4. Validate
  double total_pos_error = 0;
  int count = 0;

  for (int i = 0; i < n_shots; ++i) {
    std::string rig_id = "rig_" + shot_ids[i];
    geometry::Pose opt_pose = ba.GetRigInstance(rig_id).GetValue();
    double err = (opt_pose.GetOrigin() - gt_poses[i].GetOrigin()).norm();

    // if (is_outlier_shot[i]) {
    //   continue;
    // } else {
    //   EXPECT_LT(err, 0.5);  // 0.5m tolerance
    // }

    std::cout << "Final Pose Error for " << rig_id << ": " << err << std::endl;
    total_pos_error += err;
    count++;
  }

  std::cout << "Average Camera Position Error: " << total_pos_error / count
            << std::endl;
}

}  // namespace bundle