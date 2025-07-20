// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once 

#ifdef WITH_CUDA

// Get component sign for the direction ray
static __inline__ __device__ float3 ray_sgn(
    const float3 dir     // ray direction
) {
    return make_float3(
            signbit(dir.x) ? 1.0f : -1.0f,
            signbit(dir.y) ? 1.0f : -1.0f,
            signbit(dir.z) ? 1.0f : -1.0f);
}

static __inline__ __device__ float3 ray_flip(
    const float3 dir
) {
    return make_float3(-dir.x, -dir.y, -dir.z);
}

 static __inline__ __device__ float3 ray_invert(
    const float3 dir
) {
    // Prevent singularities
    const float eps = 1e-8;
    return make_float3(1.0 / (dir.x+eps),
                       1.0 / (dir.y+eps),
                       1.0 / (dir.z+eps));
}

// https://zhuanlan.zhihu.com/p/610258258
// Device primitive for a single ray-AABB intersection
 static __inline__ __device__ float ray_aabb(   // 计算射线起点距离与 AABB 相交点的距离。
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 sgn,    // sgn bits
    const float3 origin, // origin of aabb，    AABB的中心
    const float  r       // radius of aabb，    AABB的半径 
) {
    // From Majercik et. al 2018

    // 将 AABB中心点作为坐标系原点。
    float3 o = make_float3(query.x-origin.x, query.y-origin.y, query.z-origin.z);

    // Maximum Component
    float cmax = fmaxf(fmaxf(fabs(o.x), fabs(o.y)), fabs(o.z));    

    // If the maximum component is smaller than the radius of the AABB, then the ray origin is
    // inside the AABB; return a negative to indicate.
    float winding = cmax < r ? -1.0f : 1.0f;
    winding *= r;
    if (winding < 0) {
        return winding;
    }

    // AABB矩形中心点作为坐标系原点。
    // winding * sgn.x：根据光线方向确定要检测的AABB平面（前平面或后平面）。
    float d0 = fmaf(winding, sgn.x, - o.x) * invdir.x;      // 射线起始点o距离射线到AABB矩形的最近yz平面交点的线段长度。
    float d1 = fmaf(winding, sgn.y, - o.y) * invdir.y;      // 射线起始点o距离射线到AABB矩形的最近xz平面交点的线段长度。
    float d2 = fmaf(winding, sgn.z, - o.z) * invdir.z;      // 射线起始点o距离射线到AABB矩形的最近xy平面交点的线段长度。
    // 射线在相交平面上的投影长度。
    float ltxy = fmaf(dir.y, d0, o.y);      // 射线在AABB的"YZ平面"方向的交点的y坐标：P_y = [ O + d0 * dir ]_y，射线在AABB表面的投影。
    float ltxz = fmaf(dir.z, d0, o.z);      // 射线在AABB的"YZ平面"方向的交点的z坐标：P_z = [ O + d0 * dir ]_z
    float ltyx = fmaf(dir.x, d1, o.x);
    float ltyz = fmaf(dir.z, d1, o.z);
    float ltzx = fmaf(dir.x, d2, o.x);      // 射线在AABB的"XY平面"方向的交点的x坐标：P_x = [ O + d2 * dir ]_x，射线在AABB表面的投影。
    float ltzy = fmaf(dir.y, d2, o.y);      // 射线在AABB的"XY平面"方向的交点的y坐标：P_y = [ O + d2 * dir ]_y

    // Test hit against each plane
    bool test0 = (d0 >= 0.0f) && (fabs(ltxy) <= r) && (fabs(ltxz) <= r);
    bool test1 = (d1 >= 0.0f) && (fabs(ltyx) <= r) && (fabs(ltyz) <= r);
    bool test2 = (d2 >= 0.0f) && (fabs(ltzx) <= r) && (fabs(ltzy) <= r);

    float3 _sgn = make_float3(0.0f, 0.0f, 0.0f);

    if (test0) { _sgn.x = sgn.x; }
    else if (test1) { _sgn.y = sgn.y; }
    else if (test2) { _sgn.z = sgn.z; }

    // 射线起点与AABB矩形实际相交的点的距离d.
    float d = 0.0f;
    if (_sgn.x != 0.0f) { d = d0; } 
    else if (_sgn.y != 0.0f) { d = d1; }
    else if (_sgn.z != 0.0f) { d = d2; }
    
    if (d != 0.0f) {
        return d;
    }

    return 0.0;     // 射线与当前AABB没有交点。
    // returns: 
    //      d == 0 -> miss
    //      d >  0 -> distance
    //      d <  0 -> inside
}

static __inline__ __device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 sgn = ray_sgn(dir);
    return ray_aabb(query, dir, invdir, sgn, origin, r);
}

static __inline__ __device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 sgn = ray_sgn(dir);
    float3 invdir = ray_invert(dir);
    return ray_aabb(query, dir, invdir, sgn, origin, r);
}

static __inline__ __device__ float2 ray_aabb_with_exit(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 entry_sgn = ray_sgn(dir);
    float3 exit_sgn = ray_sgn(ray_flip(dir));
    float entry = ray_aabb(query, dir, invdir, entry_sgn, origin, r);
    float exit = ray_aabb(query, dir, invdir, exit_sgn, origin, r);
    return make_float2(entry, exit);
}

static __inline__ __device__ float2 ray_aabb_with_exit(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 invdir = ray_invert(dir);
    return ray_aabb_with_exit(query, dir, invdir, origin, r);
}

#endif //WITH_CUDA

