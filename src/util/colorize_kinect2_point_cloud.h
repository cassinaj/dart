#ifndef __COLORIZE_KIN2PTCLOUD_H__
#define __COLORIZE_KIN2PTCLOUD_H__

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

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
        uchar3 * point_color);

void convert_uchar4arr_to_uchar3arr(
        const uchar4 * uin,
        int cols,
        int rows,
		uchar3 *uout);

#endif
