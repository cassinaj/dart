
#pragma once

#include "depth_source.h"

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace dart
{
/**
 * \brief ROS sensor_msgs::Image based depth image source for DART.
 *
 * Current supported encodings:
 *  - TYPE_32FC1
 */
class RosDepthSource : public DepthSource<float, uchar3>
{
  public:
    /**
     * \brief Creates an uninitialized RosDepthSource.
     */
    RosDepthSource();

    /**
     * \brief Default destructor.
     */
    virtual ~RosDepthSource();

    /**
     * \brief Initialized the RosDepthSource, i.e. subscribes to the depth image
     * [depth_camera_topic_prefix]/[depth_camera_topic_suffix]. The intrinsic
     * parameters are obtained from the camera info topic at
     * [depth_camera_topic_prefix]/camera_info
     */
    bool initialize(const std::string& depth_camera_topic_prefix,
                    const std::string& depth_camera_topic_suffix);

    /**
     * \brief Updates the current data on host and device with the latest
     * received depth image.
     */
    void advance() override;
    const float* getDepth() const override { return depth_data_; }
    const float* getDeviceDepth() const override { return device_depth_data_; }
    const uchar3* getColor() const override { return nullptr; }
    bool hasRadialDistortionParams() const override { return false; }
    ColorLayout getColorLayout() const override { return LAYOUT_RGB; }
    void setFrame(const uint frame) override {}
    float getScaleToMeters() const override { return 1.e-3; }
    uint64_t getDepthTime() const override { return depth_time_; }
    uint64_t getColorTime() const override { return 0; }
    const sensor_msgs::CameraInfo depth_camera_info() const;

  private:
    void depth_camera_info_callback(const sensor_msgs::CameraInfo& msg);
    void depth_camera_image_callback(const sensor_msgs::Image& msg);

  private:
    std::mutex depth_camera_image_mutex_;

    /// Depth image data
    float* depth_data_;
    float* device_depth_data_;
    uint64_t depth_time_;

    /// Latest depth image received
    float* next_depth_data_;
    uint64_t next_depth_time_;

    /// ROS Subscribers
    ros::Subscriber depth_camera_image_subscriber_;

    // Camera info setup
    std::atomic<bool> depth_camera_info_available_;
    mutable std::mutex depth_camera_info_mutex_;
    sensor_msgs::CameraInfo depth_camera_info_;
};

// Inlines
inline const sensor_msgs::CameraInfo RosDepthSource::depth_camera_info() const
{
    if (!depth_camera_info_available_)
    {
        throw std::runtime_error("No camera info available.");
    }
    std::unique_lock<std::mutex> lock(depth_camera_info_mutex_);
    return depth_camera_info_;
}
}
