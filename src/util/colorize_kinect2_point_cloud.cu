#include <stdio.h>
#include "colorize_kinect2_point_cloud.h"
#include "float.h"
#include "util/dart_types.h"
#include "util/prefix.h"

#define DEPTH_Q 0.01
#define COLOR_Q 0.002199

inline PREFIX
float2 map_depth_to_color(
        const float3 point,
        const float shift_m,
        const float shift_d,
        const float2 depth_focal_length,
        const float2 color_focal_length,
        const float2 color_principal_point,
        const float * depth_to_color){
    
    float mx = point.x * depth_focal_length.x / point.z;
    float my = point.y * depth_focal_length.y / point.z;
    mx *= DEPTH_Q;
    my *= DEPTH_Q;
    
    float wx =
        (mx * mx * mx * depth_to_color[ 0]) +
        (my * my * my * depth_to_color[ 1]) +
        (mx * mx * my * depth_to_color[ 2]) +
        (mx * my * my * depth_to_color[ 3]) +
        (mx * mx * depth_to_color[ 4]) +
        (my * my * depth_to_color[ 5]) +
        (mx * my * depth_to_color[ 6]) +
        (mx * depth_to_color[ 7]) +
        (my * depth_to_color[ 8]) +
        (depth_to_color[ 9]);

    float wy =
        (mx * mx * mx * depth_to_color[10]) +
        (my * my * my * depth_to_color[11]) +
        (mx * mx * my * depth_to_color[12]) +
        (mx * my * my * depth_to_color[13]) +
        (mx * mx * depth_to_color[14]) +
        (my * my * depth_to_color[15]) +
        (mx * my * depth_to_color[16]) +
        (mx * depth_to_color[17]) +
        (my * depth_to_color[18]) +
        (depth_to_color[19]);
    
    float2 color_xy;
    color_xy.x = wx / (color_focal_length.x * COLOR_Q);
    color_xy.y = wy / (color_focal_length.y * COLOR_Q);
    color_xy.x += (shift_m / (point.z*1000)) - (shift_m / shift_d);
    
    color_xy.x = color_xy.x * color_focal_length.x + color_principal_point.x;
    color_xy.y = color_xy.y * color_focal_length.y + color_principal_point.y;
    
    return color_xy;
}

inline PREFIX
float2 map_depth_to_color(
        const float4 point,
        const float shift_m,
        const float shift_d,
        const float2 depth_focal_length,
        const float2 color_focal_length,
        const float2 color_principal_point,
        const float * depth_to_color){
    return map_depth_to_color(
            make_float3(point),
            shift_m,
            shift_d,
            depth_focal_length,
            color_focal_length,
            color_principal_point,
            depth_to_color);
}

__global__
void gpu_colorize_point_cloud(
        const float4 * points,
        int cols,
        int rows,
        float2 depth_focal_length,
        float2 depth_principal_point,
        float2 color_focal_length,
        float2 color_principal_point,
        float shift_m,
        float shift_d,
        const float * radial_distortion_params,
        const float * depth_to_color,
        const uchar4 * pixel_color,
        uchar3 * point_color){
    
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_index >= cols * rows){
        return;
    }
    
    float4 p = points[pixel_index];
    if(p.w < 0.99f){
        return;
    }
    
    float2 color_xy = map_depth_to_color(
            p,
            shift_m,
            shift_d,
            depth_focal_length,
            color_focal_length,
            color_principal_point,
            depth_to_color);
    
    if(color_xy.x >= 0. && color_xy.x < 1920 &&
            color_xy.y >= 0. && color_xy.y < 1080){
        uchar4 c = pixel_color[((int)color_xy.y) * 1920 + ((int)color_xy.x)];
        point_color[pixel_index] = make_uchar3(c.z, c.y, c.x);
    }
    else{
        point_color[pixel_index] = make_uchar3(192, 192, 192);
    }
}

void colorize_kinect2_point_cloud(
        const float4 * points,
        int cols,
        int rows,
        float2 depth_focal_length,
        float2 depth_principal_point,
        float2 color_focal_length,
        float2 color_principal_point,
        float shift_m,
        float shift_d,
        const float * radial_distortion_params,
        const float * depth_to_color,
        const uchar4 * pixel_color,
        uchar3 * point_color){
    int block = 64;
    int grid = ceil((double)rows*cols/block);
    
    gpu_colorize_point_cloud<<<grid, block>>>(
        points,
        cols,
        rows,
        depth_focal_length,
        depth_principal_point,
        color_focal_length,
        color_principal_point,
        shift_m,
        shift_d,
        radial_distortion_params,
        depth_to_color,
        pixel_color,
        point_color);
}

/////////////////// === Convert uchar4 to uchar3 array === ////////////
__global__
void gpu_convert_uchar4arr_to_uchar3arr(
        const uchar4 * uin,
        int cols,
        int rows,
		uchar3 *uout){
    
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_index >= cols * rows){
        return;
    }
    
	uchar4 c = uin[pixel_index];
	uout[pixel_index] = make_uchar3(c.x, c.y, c.z);
}

void convert_uchar4arr_to_uchar3arr(
        const uchar4 * uin,
        int cols,
        int rows,
		uchar3 *uout){
    int block = 64;
    int grid = ceil((double)rows*cols/block);
    
    gpu_convert_uchar4arr_to_uchar3arr<<<grid, block>>>(
        uin,
        cols,
        rows,
        uout);
}
