
#include "ros_depth_source.h"
#include "util/cuda_utils.h"

#include <cuda_runtime.h>
#include <sensor_msgs/image_encodings.h>
#include <string.h>
#include <vector_types.h>
#include <iostream>
#include <thread>
#include <vector>

namespace dart
{
RosDepthSource::RosDepthSource()
    : DepthSource<float, uchar3>(),
      depth_data_(nullptr),
      depth_time_(0),
      next_depth_data_(nullptr),
      next_depth_time_(0),
      scaling_factor_(1000.f),
      subsampling_factor_(1)
{
}

RosDepthSource::~RosDepthSource()
{
    delete[] depth_data_;
    cudaFree(device_depth_data_);
    delete[] next_depth_data_;
}

bool RosDepthSource::initialize(const std::string& depth_camera_topic_prefix,
                                const std::string& depth_camera_topic_suffix)
{
    ros::NodeHandle nh;

    auto depth_camera_info_subscriber =
        nh.subscribe(depth_camera_topic_prefix + "/camera_info",
                     1,
                     &RosDepthSource::depth_camera_info_callback,
                     this);

    // wait for the camera info
    ROS_INFO("Waiting for camera info ...");
    auto rate = ros::Rate(100);
    while (ros::ok() && !depth_camera_info_available_.load())
    {
        rate.sleep();
    }

    if (!depth_camera_info_available_.load())
    {
        std::cout << ">>> ROS shutdown before receiving camera info."
                  << std::endl;
        return false;
    }
    ROS_INFO("Received camera info");

    // Get camera image size and intrinsic parameters.
    _depthWidth = depth_camera_info_.width;
    _depthHeight = depth_camera_info_.height;
    _focalLength =
        make_float2(depth_camera_info_.K[0], depth_camera_info_.K[4]);
    _principalPoint =
        make_float2(depth_camera_info_.K[2], depth_camera_info_.K[5]);

    // Allocate depth image memory on host and device.
    depth_data_ = new float[_depthWidth * _depthHeight];
    next_depth_data_ = new float[_depthWidth * _depthHeight];
    memset(depth_data_, 0, _depthWidth * _depthHeight * sizeof(float));
    memset(next_depth_data_, 0, _depthWidth * _depthHeight * sizeof(float));

    cudaMalloc(&device_depth_data_, _depthWidth * _depthHeight * sizeof(float));
    advance();

    CheckCudaDieOnError();

    _hasTimestamps = true;
    _isLive = true;

    depth_time_ = 0;
    next_depth_time_ = 0;

    // Setup depth camera subscriber.
    depth_camera_image_subscriber_ = nh.subscribe(
        depth_camera_topic_prefix + "/" + depth_camera_topic_suffix,
        1,
        &RosDepthSource::depth_camera_image_callback,
        this);

    return true;
}

void RosDepthSource::advance()
{
    if (!depth_camera_info_available_.load())
    {
        // Camera info not available yet, skip callback if called too early.
        return;
    }

    // This updates the depth data with the latest received image.
    {
        std::unique_lock<std::mutex> lock(depth_camera_image_mutex_);
        if (next_depth_time_ > depth_time_)
        {
            float* tmp = depth_data_;
            depth_data_ = next_depth_data_;
            next_depth_data_ = tmp;
            depth_time_ = next_depth_time_;

            _frame++;
        }
    }

    CheckCudaDieOnError();
    // Copy data to device.
    // TODO Check why this is performed on every cycle even if no new
    // image is
    // available.
    cudaMemcpy(device_depth_data_,
               depth_data_,
               _depthWidth * _depthHeight * sizeof(float),
               cudaMemcpyHostToDevice);
    CheckCudaDieOnError();
}

void RosDepthSource::depth_camera_info_callback(
    const sensor_msgs::CameraInfo& msg)
{
    std::lock_guard<std::mutex> lock(depth_camera_info_mutex_);
    depth_camera_info_ = msg;
    depth_camera_info_available_ = true;
}

void RosDepthSource::depth_camera_image_callback(const sensor_msgs::Image& msg)
{
    if (!depth_camera_info_available_.load())
    {
        // Camera info not available yet, skip callback if called too early.
        return;
    }

    // Supported encoding:
    //  - 32FC1 (Single channel 32 bit floating point)
    if (msg.encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
        std::unique_lock<std::mutex> lock(depth_camera_image_mutex_);
        const size_t bytes = msg.data.size();
        const size_t pixels = bytes / sizeof(float);

        if (pixels != _depthWidth * _depthHeight)
        {
            throw std::runtime_error(
                "Mismatch between camera depth image float count and camera"
                "info pixels: " +
                std::to_string(pixels) + " vs " +
                std::to_string(_depthWidth * _depthHeight));
        }
        auto data = (const float*)&msg.data[0];
        for (int i = 0; i < pixels; ++i)
        {
            next_depth_data_[i] = 0.f;
        }
        for (int i = 0; i < _depthHeight; i += subsampling_factor_)
        {
            for (int j = 0; j < _depthWidth; j += subsampling_factor_)
            {
                next_depth_data_[i * _depthWidth + j] =
                    scaling_factor_ * data[i * _depthWidth + j];
            }
        }
        next_depth_time_ = msg.header.stamp.toNSec();
    }
    else
    {
        ROS_WARN_STREAM_THROTTLE(1,
                                 "Depth image encoding not supported. "
                                     << "Supported image encoding: "
                                     << "\n - TYPE_32FC1"
                                     << "\n TYPE_16UC1");
        return;
    }
}
}
