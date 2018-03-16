#ifndef K2_DEPTH_SOURCE_H
#define K2_DEPTH_SOURCE_H

#include "depth_source.h"

//#include <stdio.h>

#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <png.h>
#include <jpeglib.h>
#include <sys/stat.h>

#include "util/image_io.h"
#include "util/mirrored_memory.h"
#include "vector_types.h"

#include <chrono>
#include <thread>

#include "libfreenect2/libfreenect2.hpp"
#include "libfreenect2/frame_listener_impl.h"

unsigned char * colorPtr;

//class ColorListener : public libfreenect2::SyncMultiFrameListener {
//public:
//    ColorListener(pangolin::GlTexture * frameTex
//};

namespace dart{

template <typename DepthType, typename ColorType>
class K2DepthSource : public DepthSource<DepthType, ColorType> {
public:
    K2DepthSource();
    ~K2DepthSource();
    
    bool initialize(
            const bool do_color,
            /*
            const float2 focalLength,
            const float2 principalPoint = make_float2(0,0),
            const uint depthWidth = 512,
            const uint depthHeight = 424,
            const uint colorWidth = 1920,
            const uint colorHeight = 1080,
            */
            const float scaleToMeters = 1.f);
    
    bool hasRadialDistortionParams(){ return true; }
    
#ifdef CUDA_BUILD
    const DepthType * getDepth() const override { return _depthData->hostPtr(); }
    const DepthType * getDeviceDepth() const override { return _depthData->devicePtr(); }
    const ColorType * getColor() const override { return _colorData->hostPtr(); }
    const ColorType * getDeviceColor() const { return _colorData->devicePtr(); }
#else
    const DepthType * getDepth() const override { return _depthData; }
    const DepthType * getDeviceDepth() const override { return 0; }
    const ColorType * getColor() const override { return _colorData; }
    const ColorType * getDeviceColor() const { return 0; }
#endif
    
    uint64_t getDepthTime() const { return _depthTimes[this->_frame]; }
    
    void setFrame(const uint frame);
    
    void advance();
    
    bool hasRadialDistortionParams() const { return true; }
    float2 getFocalLength() const {return this->_focalLength;}
    float2 getPrincipalPoint() const {return this->_principalPoint;}
    float2 getColorFocalLength() const {return _color_focal_length;}
    float2 getColorPrincipalPoint() const {return _color_principal_point;}
    float getShiftM() const {return _shift_m;}
    float getShiftD() const {return _shift_d;}

#ifdef CUDA_BUILD
    float * getHostDepthToColor() const {return _depthToColor->hostPtr();}
    float * getDeviceDepthToColor() const {return _depthToColor->devicePtr();}
    const float * getRadialDistortionParams() const {
        return _depth_radial_distortion->hostPtr();
    }
    const float * getDeviceRadialDistortionParams() const {
        return _depth_radial_distortion->devicePtr();
    }
#else
    float * getHostDepthToColor() const {return _depthToColor;}
    float * getDeviceDepthToColor() const {return 0;}
    const float * getRadialDistortionParams() const {
        return _depth_radial_distortion;
    }
    const float * getDeviceRadialDistortionParams() const {return 0;}
#endif
    
    float getScaleToMeters() const { return _scaleToMeters; }

private:
    
    void readDepth();

#ifdef CUDA_BUILD
    MirroredVector<DepthType> * _depthData;
    MirroredVector<ColorType> * _colorData;
    MirroredVector<float> * _depthToColor;
    MirroredVector<float> * _depth_radial_distortion;
#else
    DepthType * _depthData;
    ColorType * _colorData;
    float _depthToColor[20];
    float _depth_radial_distortion[5];
#endif
    
    bool _do_color;
    uint _firstDepthFrame;
    uint _lastDepthFrame;
    
    float2 _color_focal_length;
    float2 _color_principal_point;
    float _shift_d, _shift_m;
    //float _depth_to_color[20];
    float _scaleToMeters;
    std::vector<ulong> _depthTimes;
    libfreenect2::Freenect2 _freenect2;
    libfreenect2::PacketPipeline * _pipeline;
    libfreenect2::Freenect2Device * _device;
    libfreenect2::SyncMultiFrameListener * _listener;
    bool _initialized;
};

// Implementation
template <typename DepthType, typename ColorType>
K2DepthSource<DepthType,ColorType>::K2DepthSource() :
    DepthSource<DepthType, ColorType>(),
    _firstDepthFrame(0),
    _depthData(0),
    _colorData(0) {}

template <typename DepthType, typename ColorType>
K2DepthSource<DepthType, ColorType>::~K2DepthSource() {
#ifdef CUDA_BUILD
    delete _depthData;
    delete _colorData;
#else
    delete [] _depthData;
    delete [] _colorData;
#endif
    
    _device->stop();
    _device->close();
    
    // crashy crash?
    //delete _pipeline;
    //delete _listener;
    //delete _device;
}

template <typename DepthType, typename ColorType>
bool K2DepthSource<DepthType, ColorType>::initialize(
        bool do_color,
        /*
        const float2 focalLength,
        const float2 principalPoint,
        const uint depthWidth,
        const uint depthHeight,
        const uint colorWidth,
        const uint colorHeight,
        */
        const float scaleToMeters){
    
    this->_do_color = do_color;
    this->_frame = 0;
    /*
    this->_focalLength = focalLength;
    */
    this->_depthWidth = 512;//depthWidth;
    this->_depthHeight = 424;//depthHeight;
    this->_colorWidth = 1920;//colorWidth;
    this->_colorHeight = 1080;//colorHeight;
    
    /*
    if(principalPoint.x == 0){
        this->_principalPoint = make_float2(
                this->_depthWidth/2, this->_depthHeight/2);
    }
    else{
        this->_principalPoint = principalPoint;
    }
    */
    _scaleToMeters = scaleToMeters;
        
    int nDevices = _freenect2.enumerateDevices();
    std::cout << "found " << nDevices << " devices" << std::endl;
    for(int i = 0; i < nDevices; ++i){
        std::cout << "device " << i << " has serial number "
                << _freenect2.getDeviceSerialNumber(i) << std::endl;
    }
    
    if(nDevices == 0){
        std::cerr << "could not find any devices" << std::endl;
        _initialized = false;
        return false;
    }
    
    std::cout << "attempting to open device 0" << std::endl;
    
    _pipeline = new libfreenect2::OpenCLPacketPipeline();
    _device = _freenect2.openDevice(0, _pipeline);
    
    if(_device == NULL){
        std::cerr << "could not open device" << std::endl;
        _initialized = false;
        return false;
    }
    
    std::cout << "opened device has serial number "
            << _device->getSerialNumber() << std::endl;
    
    std::cout << "and firmware version " << _device->getFirmwareVersion()
            << std::endl;
    
    _initialized = true;
    
    printf("INITIALIZING TO SIZE %i\n",
            this->_depthWidth * this->_depthHeight);
#ifdef CUDA_BUILD
    _depthData = new MirroredVector<DepthType>(
            this->_depthWidth * this->_depthHeight);
    if(do_color){
        _colorData = new MirroredVector<ColorType>(
            this->_colorWidth * this->_colorHeight);
    }
    _depthToColor = new MirroredVector<float>(20);
    _depth_radial_distortion = new MirroredVector<float>(5);
#else
    _depthData = new DepthType[this->_depthWidth*this->_depthHeight];
    if(do_color){
        _colorData = new ColorType[this->_colorWidth*this->_colorHeight];
    }
#endif
    
    _listener = new libfreenect2::SyncMultiFrameListener(
            libfreenect2::Frame::Depth | libfreenect2::Frame::Color);
    _device->setColorFrameListener(_listener);
    _device->setIrAndDepthFrameListener(_listener);
    _device->start();
    
    libfreenect2::Freenect2Device::IrCameraParams ir_params =
            _device->getIrCameraParams();
    this->_focalLength.x = -ir_params.fx;
    this->_focalLength.y = ir_params.fy;
    this->_principalPoint.x = ir_params.cx;
    this->_principalPoint.y = ir_params.cy;
    
    libfreenect2::Freenect2Device::ColorCameraParams color_params =
            _device->getColorCameraParams();
    _color_focal_length.x = color_params.fx;
    _color_focal_length.y = color_params.fy;
    _color_principal_point.x = color_params.cx;
    _color_principal_point.y = color_params.cy;
    _shift_d = color_params.shift_d;
    _shift_m = color_params.shift_m;
    
#ifdef CUDA_BUILD
    _depthToColor->hostPtr()[ 0] = color_params.mx_x3y0;
    _depthToColor->hostPtr()[ 1] = color_params.mx_x0y3;
    _depthToColor->hostPtr()[ 2] = color_params.mx_x2y1;
    _depthToColor->hostPtr()[ 3] = color_params.mx_x1y2;
    _depthToColor->hostPtr()[ 4] = color_params.mx_x2y0;
    _depthToColor->hostPtr()[ 5] = color_params.mx_x0y2;
    _depthToColor->hostPtr()[ 6] = color_params.mx_x1y1;
    _depthToColor->hostPtr()[ 7] = color_params.mx_x1y0;
    _depthToColor->hostPtr()[ 8] = color_params.mx_x0y1;
    _depthToColor->hostPtr()[ 9] = color_params.mx_x0y0;
    _depthToColor->hostPtr()[10] = color_params.my_x3y0;
    _depthToColor->hostPtr()[11] = color_params.my_x0y3;
    _depthToColor->hostPtr()[12] = color_params.my_x2y1;
    _depthToColor->hostPtr()[13] = color_params.my_x1y2;
    _depthToColor->hostPtr()[14] = color_params.my_x2y0;
    _depthToColor->hostPtr()[15] = color_params.my_x0y2;
    _depthToColor->hostPtr()[16] = color_params.my_x1y1;
    _depthToColor->hostPtr()[17] = color_params.my_x1y0;
    _depthToColor->hostPtr()[18] = color_params.my_x0y1;
    _depthToColor->hostPtr()[19] = color_params.my_x0y0;
    _depthToColor->syncHostToDevice();
    _depth_radial_distortion->hostPtr()[0] = ir_params.k1;
    _depth_radial_distortion->hostPtr()[1] = ir_params.k2;
    _depth_radial_distortion->hostPtr()[2] = ir_params.p1;
    _depth_radial_distortion->hostPtr()[3] = ir_params.p2;
    _depth_radial_distortion->hostPtr()[4] = ir_params.k3;
    _depth_radial_distortion->syncHostToDevice();
#else
    _depthToColor[ 0] = color_params.mx_x3y0;
    _depthToColor[ 1] = color_params.mx_x0y3;
    _depthToColor[ 2] = color_params.mx_x2y1;
    _depthToColor[ 3] = color_params.mx_x1y2;
    _depthToColor[ 4] = color_params.mx_x2y0;
    _depthToColor[ 5] = color_params.mx_x0y2;
    _depthToColor[ 6] = color_params.mx_x1y1;
    _depthToColor[ 7] = color_params.mx_x1y0;
    _depthToColor[ 8] = color_params.mx_x0y1;
    _depthToColor[ 9] = color_params.mx_x0y0;
    _depthToColor[10] = color_params.my_x3y0;
    _depthToColor[11] = color_params.my_x0y3;
    _depthToColor[12] = color_params.my_x2y1;
    _depthToColor[13] = color_params.my_x1y2;
    _depthToColor[14] = color_params.my_x2y0;
    _depthToColor[15] = color_params.my_x0y2;
    _depthToColor[16] = color_params.my_x1y1;
    _depthToColor[17] = color_params.my_x1y0;
    _depthToColor[18] = color_params.my_x0y1;
    _depthToColor[19] = color_params.my_x0y0;
    _depth_radial_distortion[0] = ir_params.k1;
    _depth_radial_distortion[1] = ir_params.k2;
    _depth_radial_distortion[2] = ir_params.p1;
    _depth_radial_distortion[3] = ir_params.p2;
    _depth_radial_distortion[4] = ir_params.k3;
#endif

    this->_isLive = true;
    return true;
}

template <typename DepthType, typename ColorType>
void K2DepthSource<DepthType, ColorType>::setFrame(const uint frame){
    readDepth();
}

template <typename DepthType, typename ColorType>
void K2DepthSource<DepthType, ColorType>::advance() {
    readDepth();
}

template <typename DepthType, typename ColorType>
void K2DepthSource<DepthType, ColorType>::readDepth() {
    libfreenect2::FrameMap frameMap;
    //if(_listener.hasNewFrame()){
    //    libfreenect2::FrameMap frameMap;
    //    _listener
    //}
    //printf("A\n");
    _listener->waitForNewFrame(frameMap);
    //printf("B\n");
    libfreenect2::Frame * depthFrame = frameMap[libfreenect2::Frame::Depth];
    //std::cout << "depth@ " << (depthFrame->timestamp) << std::endl;
    //DepthType cpoint = (*(this->_depthData + sizeof(MirroredVector<float>) * (this->_depthHeight / 2 * this->_depthWidth + this->_depthWidth / 2))->hostPtr());
    //std::cout << "*********central point " << cpoint;//.x << " " << cpoint.y << " " << cpoint.z << " " << "\n";

    cudaMemcpy(
            _depthData->devicePtr(),
            depthFrame->data,
            this->_depthWidth * this->_depthHeight * sizeof(DepthType),
            cudaMemcpyHostToDevice);
    _depthData->syncDeviceToHost();
    
    // _depthData
    //FILE * frameFile = fopen("frame.dat","wb");
    //fwrite(_depthData, sizeof(DepthType), this->_depthWidth * this->_depthHeight, frameFile);
    //fclose(frameFile);

    //std::cout << "testing";
    //ROS_INFO("testing-ros");

    if(_do_color){
        libfreenect2::Frame * colorFrame = frameMap[libfreenect2::Frame::Color];
        cudaMemcpy(
                _colorData->devicePtr(),
                colorFrame->data,
                this->_colorWidth * this->_colorHeight * sizeof(ColorType),
                cudaMemcpyHostToDevice);
        _colorData->syncDeviceToHost();
    }
    
    _listener->release(frameMap);
}

};

#endif
