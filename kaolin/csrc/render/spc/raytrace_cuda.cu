// Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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

#define CUB_NS_PREFIX \
  namespace kaolin    \
  {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <stdio.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#ifdef EXPERIMENTAL
#include <ATen/native/cuda/KernelUtils.cuh>
#else
#include <THC/THCAtomics.cuh>
#endif
// TODO(ttakikawa): newer versions of PyTorch will migrate to <ATen/cuda/Atomics.cuh>.
// How do we manage these dependencies?

#define CUB_STDERR
#include <cub/device/device_scan.cuh>

#include "../../spc_math.h"
#include "../../spc_utils.cuh"
#include "spc_render_utils.cuh"

namespace kaolin
{

  using namespace at::indexing;

#define RT_NUM_THREADS 1024

  ////////////////////////////////////////////////////////////////////////////////////////////////
  /// Constants
  ////////////////////////////////////////////////////////////////////////////////////////////////

  __constant__ uint8_t VOXEL_ORDER[8][8] = {
      {0, 1, 2, 4, 3, 5, 6, 7},
      {1, 0, 3, 5, 2, 4, 7, 6},
      {2, 0, 3, 6, 1, 4, 7, 5},
      {3, 1, 2, 7, 0, 5, 6, 4},
      {4, 0, 5, 6, 1, 2, 7, 3},
      {5, 1, 4, 7, 0, 3, 6, 2},
      {6, 2, 4, 7, 0, 3, 5, 1},
      {7, 3, 5, 6, 1, 2, 4, 0}};

  ////////////////////////////////////////////////////////////////////////////////////////////////
  /// Kernels
  ////////////////////////////////////////////////////////////////////////////////////////////////

  // This function will initialize the nuggets array with each ray pointing to the octree root
  __global__ void
  init_nuggets_cuda_kernel(
      const uint num,
      uint2 *nuggets)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      nuggets[tidx].x = tidx; // ray idx
      nuggets[tidx].y = 0;    // point idx
    }
  }

  // 使用AABB判断第 tidx 个射线是否与当前节点相交，如果相交则记录该节点的叶子节点数量到 info[tidx] = __popc(octree[pidx])
  __global__ void
  decide_cuda_kernel(
      const uint num,
      // https://kaolin.readthedocs.io/en/stable/notes/spc_summary.html#spc-points
      // https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html#spc-attributes，其中的3D和2D 例子。
      const point_data *__restrict__ points,    // 层次化节点，其中根节点为[0,0,0]。叶子节点坐标在[0, 2^max_level-1]之间。
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      const uint2 *__restrict__ nuggets,
      uint *__restrict__ info,
      // torch.ByteTensor是 8 位无符号整数（uint8），标记体素是否被占用（0 表示空，1 表示占用）。
      const uint8_t *__restrict__ octree,   
      const uint32_t level,
      const uint32_t not_done)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      uint ridx = nuggets[tidx].x;
      uint pidx = nuggets[tidx].y;
      point_data p = points[pidx];
      float3 o = ray_o[ridx];
      float3 d = ray_d[ridx];

      // Radius of voxel
      float r = 1.0 / ((float)(0x1 << level));      // level的值由 0 逐步增加到 target_level-1 最大

      // Transform to [-1, 1]
      const float3 vc = make_float3(
          fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),      // 因为随着level增加，节点整数坐标p的范围也增加，0 到 2^L - 1）
          fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
          fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

      // Compute aux info (precompute to optimize)
      float3 sgn = ray_sgn(d);
      float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
      
      // 计算射线起点 与 半径为r的栅格的表面交点的距离。
      float depth = ray_aabb(o, d, ray_inv, sgn, vc, r);    // r 为第 level 层八叉树的半径。 半径 r 由粗（大）到细（小：最底层）。

      if (not_done)
      {// 未到达叶子节点时

        if (depth != 0.0) // 射线与当前节点的AABB有交点。
          info[tidx] = __popc(octree[pidx]);      // 保存该节点的子节点数量。
        else  // 没有交点。
          info[tidx] = 0;
      }
      else
      {// 到达叶子节点(底部)时

        if (depth > 0.0)    // 与叶子节点相交。
          info[tidx] = 1;
        else  // 没有交点
          info[tidx] = 0;
      }
    }
  }

  // Overloaded version of function above that returns depth of voxel/ ray entry points
  __global__ void
  decide_cuda_kernel(
      const uint num,
      const point_data *__restrict__ points,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      const uint2 *__restrict__ nuggets,
      float *depth,
      uint *__restrict__ info,
      const uint8_t *__restrict__ octree,
      const uint32_t level)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      uint ridx = nuggets[tidx].x;
      uint pidx = nuggets[tidx].y;
      point_data p = points[pidx];
      float3 o = ray_o[ridx];
      float3 d = ray_d[ridx];

      // Radius of voxel
      float r = 1.0 / ((float)(0x1 << level));

      // Transform to [-1, 1]
      const float3 vc = make_float3(
          fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),
          fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
          fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

      // Compute aux info (precompute to optimize)
      float3 sgn = ray_sgn(d);
      float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
      
      // 计算射线ray_o[ridx]起点与AABB包围盒的交点的距离值。
      depth[tidx] = ray_aabb(o, d, ray_inv, sgn, vc, r);

      // Perform AABB check
      if (depth[tidx] > 0.0)
      {
        info[tidx] = 1; // mark to keep
      }
      else
      {
        info[tidx] = 0;
      }
    }
  }

  // Overloaded version of function above that returns depth of voxel/ ray entry and exit points
  // 返回： 射线是否对应根节点 info[tidx] ， 射线起点到根节点的距离 depth[tidx] 。
  __global__ void
  decide_cuda_kernel(
      const uint num,
      const point_data *__restrict__ points,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      const uint2 *__restrict__ nuggets,
      float2 *__restrict__ depth,
      uint *__restrict__ info,
      // torch.ByteTensor是 8 位无符号整数（uint8），标记体素是否被占用（0 表示空，1 表示占用）。
      const uint8_t *__restrict__ octree,   
      const uint32_t level)
  {
    // 第 tidx 个（节点全局序号 ， 射线序号）对 。
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)   // num 为 （节点全局序号 ， 射线序号）对 的数量。
    {
      uint ridx = nuggets[tidx].x;  // 射线序号
      uint pidx = nuggets[tidx].y;  // 节点全局序号
      point_data p = points[pidx];
      float3 o = ray_o[ridx];
      float3 d = ray_d[ridx];

      // Radius of voxel
      float r = 1.0 / ((float)(0x1 << level));

      // Transform to [-1, 1]
      const float3 vc = make_float3(
          fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),
          fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
          fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

      // Compute aux info (precompute to optimize)
      float3 sgn = ray_sgn(d);
      float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
      
      // 射线起点与根节点的穿入点距离 和 穿出点距离
      depth[tidx] = ray_aabb_with_exit(o, d, ray_inv, vc, r);

      // Perform AABB check
      if (depth[tidx].x > 0.0 && depth[tidx].y > 0.0)
      {
        info[tidx] = 1; // mark to keep
      }
      else
      {
        info[tidx] = 0;
      }
    }
  }

  // 返回所有 ("y"子节点全局序号， "x"射线序号) 对 <————> "nuggets_out"
  __global__ void
  subdivide_cuda_kernel(
      const uint num,                           // 射线数量。
      const uint2 *__restrict__ nuggets_in,     // 射线序号 ——> 当前层相交的节点序号
      // 用于保存射线在下一层节点中的实际相交的节点序号。
      uint2 *__restrict__ nuggets_out,
      const float3 *__restrict__ ray_o,         // 射线起点（相机光心坐标）。坐标取值在[-1,1]之间。
      // https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html#spc-attributes，其中的3D和2D 例子。
      // 从八叉树中生成的点云 kaolin::generate_points_cuda ，生成的点云是一个层次化的点云[N, 3]。随着层次加深，量化的坐标范围越大，初始值为[0, 1]
      const point_data *__restrict__ points,    
      const uint8_t *__restrict__ octree,
      const uint *__restrict__ exclusive_sum,   // 用于八叉叶子节点的快速索引。存储了每个节点在扁平化存储中的偏移量。
      const uint *__restrict__ info,            // 射线对应的节点数量。
      const uint *__restrict__ prefix_sum,      // info的包含前缀和。
      const uint32_t level)                     // octree 的当前节点的分辨率等级
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])   // 判断该射线是否有需要处理的节点。
    {
      uint ridx = nuggets_in[tidx].x;
      // 射线对应的《父节点索引》
      int pidx = nuggets_in[tidx].y;
      // 父节点的坐标
      point_data p = points[pidx];            

      uint base_idx = prefix_sum[tidx];

      // **********************************
      // 该父节点字节对应的子节点情况 "o" 。
      uint8_t o = octree[pidx];       
      // 该父节点的全局索引 "s" 。
      uint s = exclusive_sum[pidx];
      // **********************************

      // 该父节点所在层的网格半径大小 "1/2^level" 。
      float scale = 1.0 / ((float)(0x1 << level));
      float3 org = ray_o[ridx];
      
      // 计算射线起点相对父节点中心坐标的坐标。
      float x = (0.5f * org.x + 0.5f) - scale * ((float)p.x + 0.5);   // (0.5f * org.x + 0.5f) 将射线起点坐标从【-1,1】缩放到【0，1】
      float y = (0.5f * org.y + 0.5f) - scale * ((float)p.y + 0.5);   // 当前父节点的中心坐标 "p+0.5" 。
      float z = (0.5f * org.z + 0.5f) - scale * ((float)p.z + 0.5);   // 乘以 scale 的目的值始终将每个层级的节点缩放到【0,1】之间。（因为随着level增加，节点整数坐标范围也增加）

      // 射线（ray）起点相对节点的方向会影响子节点的遍历顺序。
      // 这是因为射线的传播方向决定了它与子节点的相交优先级。
      // 为了提高效率，遍历子节点的顺序需要根据射线的前进方向进行优化，以《优先访问最先相交的子节点》。
      // 为什么需要优化遍历顺序：
      // 1、减少不必要的测试：如果射线在某个子节点中已经找到相交点，后续子节点的检测可能就不需要了。通过优先访问最有可能相交的子节点，可以尽早终止遍历。
      // 2、提升性能：在实时渲染或大规模场景中，减少相交测试的次数对性能提升至关重要。
      uint code = 0;
      if (x > 0)
        code = 4;
      if (y > 0)
        code += 2;
      if (z > 0)
        code += 1;
      
      // 再遍历实际相交的子节点的下一层8个子节点，判断第二层的节点在octree中是否存在。
      for (uint i = 0; i < 8; i++)
      {
        // VOXEL_ORDER 定义了根据射线方向遍历子节点的顺序。不同的射线方向会导致不同的子节点优先级。
        uint j = VOXEL_ORDER[code][i];    
        
        if (o & (0x1 << j)) // 检查当前层节点o的第j个子节点是否存在。
        {
          // ***************************************
          // cnt: 子节点相对父节点的偏移量。
          uint cnt = __popc(o & ((0x2 << j) - 1)); 
          // 子节点的全局序号。
          nuggets_out[base_idx].y = s + cnt;    
          // 对应的射线序号。
          nuggets_out[base_idx++].x = ridx;     
          // ***************************************
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  mark_pack_boundaries_cuda_kernel(
      const int64_t num,
      const scalar_t *__restrict__ pack_ids,
      uint *__restrict__ boundaries)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      if (tidx == 0)
      {
        boundaries[tidx] = 1;
      }
      else
      {
        boundaries[tidx] = pack_ids[tidx - 1] == pack_ids[tidx] ? 0 : 1;
      }
    }
  }

  // This function will take a buffer and remove the zero pads
  template <typename scalar_t>
  __global__ void
  compactify_cuda_kernel(
      const uint num,
      const scalar_t *__restrict__ buffer_in,
      scalar_t *__restrict__ buffer_out,
      const uint *__restrict__ info,
      const uint *__restrict__ prefix_sum)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
    {
      buffer_out[prefix_sum[tidx]] = buffer_in[tidx];
    }
  }

  template <typename scalar_t>
  __global__ void
  diff_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      const int64_t *__restrict__ pack_indices)
  {

    int64_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int64_t upper_bound = (tidx == num_packs - 1) ? num_feats : pack_indices[tidx + 1];
      for (int64_t i = pack_indices[tidx]; i < upper_bound - 1; ++i)
      {
        for (int64_t j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] = feats_in[(i + 1) * feat_dim + j] - feats_in[i * feat_dim + j];
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  sum_reduce_cuda_kernel(
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      const int32_t *__restrict__ inclusive_sum)
  {

    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_feats)
    {
      for (int i = 0; i < feat_dim; ++i)
      {
        int idx = (inclusive_sum[tidx] - 1) * feat_dim + i;
#ifdef EXPERIMENTAL
        int numel = num_feats * feat_dim;
        at::native::fastAtomicAdd(feats_out, idx, numel, feats_in[tidx * feat_dim + i], true);
#else
        gpuAtomicAdd(feats_out + idx, feats_in[tidx * feat_dim + i]);
#endif
      }
    }
  }

  // This kernel is the same as sum_reduce but avoids atomic add by packing the ops.
  // It however will cause thread divergence.
  template <typename scalar_t>
  __global__ void
  packed_sum_reduce_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      const int64_t *__restrict__ pack_indices)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int64_t upper_bound = (tidx == num_packs - 1) ? num_feats * feat_dim : pack_indices[tidx + 1];
      for (int i = pack_indices[tidx]; i < upper_bound - 1; ++i)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] += feats_in[i * feat_dim + j];
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  cumprod_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      int32_t *__restrict__ pack_indices, // maps idx of pack -> beginning of global idx
      int32_t offset)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int upper_bound = (tidx == num_packs - 1) ? num_feats : pack_indices[tidx + 1];
      int begin = pack_indices[tidx];
      if (offset == 0)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
      }
      for (int i = begin + 1; i < upper_bound; ++i)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] = feats_in[(i - offset) * feat_dim + j] * feats_out[(i - 1) * feat_dim + j];
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  cumprod_reverse_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      int32_t *__restrict__ pack_indices, // maps idx of pack -> beginning of global idx
      int32_t offset)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int upper_bound = (tidx == num_packs - 1) ? num_feats : pack_indices[tidx + 1];
      int begin = pack_indices[tidx];
      if (offset == 0)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[(upper_bound - 1) * feat_dim + j] = feats_in[(upper_bound - 1) * feat_dim + j];
        }
      }
      for (int i = upper_bound - 2; i >= begin; --i)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] = feats_in[(i + offset) * feat_dim + j] * feats_out[(i + 1) * feat_dim + j];
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  cumsum_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      int32_t *__restrict__ pack_indices, // maps idx of pack -> beginning of global idx
      int32_t offset)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int upper_bound = (tidx == num_packs - 1) ? num_feats : pack_indices[tidx + 1];
      int begin = pack_indices[tidx];
      if (offset == 0)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
      }
      for (int i = begin + 1; i < upper_bound; ++i)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] = feats_in[(i - offset) * feat_dim + j] + feats_out[(i - 1) * feat_dim + j];
        }
      }
    }
  }

  template <typename scalar_t>
  __global__ void
  cumsum_reverse_cuda_kernel(
      const int64_t num_packs,
      const int64_t num_feats,
      const int64_t feat_dim,
      const scalar_t *__restrict__ feats_in,
      scalar_t *__restrict__ feats_out,
      int32_t *__restrict__ pack_indices, // maps idx of pack -> beginning of global idx
      int32_t offset)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < num_packs)
    {
      int upper_bound = (tidx == num_packs - 1) ? num_feats : pack_indices[tidx + 1];
      int begin = pack_indices[tidx];
      if (offset == 0)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[(upper_bound - 1) * feat_dim + j] = feats_in[(upper_bound - 1) * feat_dim + j];
        }
      }
      for (int i = upper_bound - 2; i >= begin; --i)
      {
        for (int j = 0; j < feat_dim; ++j)
        {
          feats_out[i * feat_dim + j] = feats_in[(i + offset) * feat_dim + j] + feats_out[(i + 1) * feat_dim + j];
        }
      }
    }
  }

  //返回： 「根节点的全局序号」 和 「对应的射线序号」对 保存在新的 nuggets 中。以及射线起点到相交节点的穿入和穿出点的距离值。
  std::vector<at::Tensor> raytrace_cuda_impl(
      at::Tensor octree,           // torch.ByteTensor是 8 位无符号整数（uint8），标记体素是否被占用（0 表示空，1 表示占用）。
      // https://kaolin.readthedocs.io/en/stable/notes/spc_summary.html#spc-points
      // https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html#spc-attributes，其中的3D和2D 例子。
      at::Tensor points,  // 从八叉树中生成的点云 kaolin::generate_points_cuda ，生成的点云是一个层次化的点云[N, 3]。其中根节点为[0,0,0]。
      at::Tensor pyramid,
      at::Tensor exclusive_sum,    // 用于八叉叶子节点的快速索引。存储了每个节点在扁平化存储中的偏移量。
      at::Tensor ray_o,            // 射线起点（相机光心坐标）。坐标取值在[-1,1]之间。
      at::Tensor ray_d,            // 每个点云的方向向量。
      uint32_t max_level,
      uint32_t target_level,
      bool return_depth,
      bool with_exit)
  {

    uint num = ray_o.size(0);

    uint8_t *octree_ptr = octree.data_ptr<uint8_t>();
    // 从八叉树中生成的点云 kaolin::generate_points_cuda ，生成的点云是一个层次化的点云[N, 3]。
    // 其中根节点为[0,0,0]。
    point_data *points_ptr = reinterpret_cast<point_data *>(points.data_ptr<short>());    
    uint *exclusive_sum_ptr = reinterpret_cast<uint *>(exclusive_sum.data_ptr<int>());
    float3 *ray_o_ptr = reinterpret_cast<float3 *>(ray_o.data_ptr<float>());
    float3 *ray_d_ptr = reinterpret_cast<float3 *>(ray_d.data_ptr<float>());

    // allocate local GPU storage
    at::Tensor nuggets0 = at::empty({num, 2}, octree.options().dtype(at::kInt));
    at::Tensor nuggets1;

    uint depth_dim = with_exit ? 2 : 1;
    at::Tensor depths0;
    at::Tensor depths1;

    // Generate proposals (first proposal is root node)
    // // 序号为 0 时对应的是八叉树的根节点。在 SPC 的实现中，根节点通常表示整个空间的起点，其坐标在整数坐标系中通常是 (0, 0, 0)。
    init_nuggets_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
        num, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()));

    // cnt 保存当前 nuggets0 节点中的下一层子节点的总数量。
    uint cnt, buffer = 0;

    for (uint32_t l = 0; l <= target_level; l++)
    {
      
      // info 用于记录每个节点对应的子节点数量。
      at::Tensor info = at::empty({num + 1}, octree.options().dtype(at::kInt));
      uint *info_ptr = reinterpret_cast<uint *>(info.data_ptr<int>());    

      /// ############################################### Step1: ###############################################
      if (l == target_level && return_depth)
      {
        depths0 = at::empty({num, depth_dim}, octree.options().dtype(at::kFloat));

        if (with_exit)
        {
          // 返回： 射线是否对应根节点 info[tidx] ，是则为1，否则为0， 射线起点到根节点的穿入点和穿出点的距离 depth[tidx] 。
          decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
              num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()),
              reinterpret_cast<float2 *>(l == target_level ? depths0.data_ptr<float>() : 0), info_ptr, octree_ptr, l);
        }
        else
        {
          // 返回： 射线是否对应根节点 info[tidx] ， 射线起点到根节点的穿入点的距离 depth[tidx] 。
          decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
              num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()),
              l == target_level ? depths0.data_ptr<float>() : 0, info_ptr, octree_ptr, l);
        }
      }
      else
      {
        // ******************************************************
        // 统计： 返回 info 数组中保存每个射线相交的节点的子节点数量。
        // ******************************************************
        decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
            num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()),
            info_ptr, octree_ptr, l, target_level - l);
      }

      /// ############################################### Step2: ###############################################

      // 创建数组，用于记录 info 数组中的前缀和。
      at::Tensor prefix_sum = at::empty({num + 1}, octree.options().dtype(at::kInt));
      uint *prefix_sum_ptr = reinterpret_cast<uint *>(prefix_sum.data_ptr<int>());

      // set first element to zero， 将前缀和数组的第一个元素赋值为0.
      CubDebugExit(cudaMemcpy(prefix_sum_ptr, &buffer, sizeof(uint), cudaMemcpyHostToDevice));

      // set up memory for DeviceScan calls
      void *temp_storage_ptr = NULL;
      // 计算存储所需要的空间。
      uint64_t temp_storage_bytes = get_cub_storage_bytes(
          temp_storage_ptr, info_ptr, prefix_sum_ptr, num + 1);
      
      // 分配临时存储。 
      at::Tensor temp_storage = at::empty({(int64_t)temp_storage_bytes}, octree.options());
      temp_storage_ptr = (void *)temp_storage.data_ptr<uint8_t>();
      
      // 计算 info_ptr 数组的前缀和。
      CubDebugExit(cub::DeviceScan::InclusiveSum(temp_storage_ptr,  temp_storage_bytes,  info_ptr,  prefix_sum_ptr + 1, num)); // start sum on second element
      
      // *********************************************
      // 得到所有射线在下一层节点的候选相交的总数量 cnt 。
      // *********************************************
      cudaMemcpy(&cnt, prefix_sum_ptr + num, sizeof(uint), cudaMemcpyDeviceToHost);
      // 长度为当前层所有候选子节点的数量。
      nuggets1 = at::empty({cnt, 2}, octree.options().dtype(at::kInt));

      if (cnt == 0)
      {
        num = 0;
        nuggets0 = nuggets1;
        if (return_depth)
          depths1 = at::empty({0, depth_dim}, octree.options().dtype(at::kFloat));
        break;
      }

      /// ############################################### Step3: #########################################################
      if (l < target_level)
      {
        // *******************************************************************
        // 将 「子节点的全局序号」 和 「对应的射线序号」对 保存在新的 nuggets1 中。
        // *******************************************************************
        subdivide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
            num, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()), reinterpret_cast<uint2 *>(nuggets1.data_ptr<int>()), ray_o_ptr, points_ptr,
            octree_ptr, exclusive_sum_ptr, info_ptr, prefix_sum_ptr, l);
      }
      else
      {
        compactify_cuda_kernel<uint2><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
            num, reinterpret_cast<uint2 *>(nuggets0.data_ptr<int>()), reinterpret_cast<uint2 *>(nuggets1.data_ptr<int>()),
            info_ptr, prefix_sum_ptr);
        if (return_depth)
        {
          depths1 = at::empty({cnt, depth_dim}, octree.options().dtype(at::kFloat));

          if (with_exit)
          {
            compactify_cuda_kernel<float2><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, reinterpret_cast<float2 *>(depths0.data_ptr<float>()),
                reinterpret_cast<float2 *>(depths1.data_ptr<float>()),
                info_ptr, prefix_sum_ptr);
          }
          else
          {
            compactify_cuda_kernel<float><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, depths0.data_ptr<float>(), depths1.data_ptr<float>(),
                info_ptr, prefix_sum_ptr);
          }
        }
      }

      //////////////////////////////////////
      // 准备计算下一层节点，重复以上步骤。
      //////////////////////////////////////
      nuggets0 = nuggets1;
      num = cnt;
    }

    if (return_depth)
    { //返回： 「根节点的全局序号」 和 「对应的射线序号」对 保存在新的 nuggets 中。以及射线起点到相交节点的穿入和穿出点的距离值。
      return {nuggets0.index({Slice(0, num)}).contiguous(),
              depths1.index({Slice(0, num)}).contiguous()};
    }
    else
    { //返回： 「根节点的全局序号」 和 「对应的射线序号」对 保存在新的 nuggets 中。
      return {nuggets0.index({Slice(0, num)}).contiguous()};
    }
  }

  void mark_pack_boundaries_cuda_impl(
      at::Tensor pack_ids,
      at::Tensor boundaries)
  {
    int64_t num = pack_ids.size(0);
    AT_DISPATCH_INTEGRAL_TYPES(pack_ids.scalar_type(), "mark_pack_boundaries_cuda", ([&]
                                                                                     {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(boundaries));
        auto stream = at::cuda::getCurrentCUDAStream();
        mark_pack_boundaries_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS, 0, stream>>>(
            num,
            pack_ids.data_ptr<scalar_t>(),
            reinterpret_cast<uint*>(boundaries.data_ptr<int>())); }));
  }

  void diff_cuda_impl(
      int64_t num_packs,
      int64_t num_feats,
      int64_t feat_dim,
      at::Tensor feats_in,
      at::Tensor feats_out,
      at::Tensor pack_indices)
  {

    int64_t *pack_indices_ptr = pack_indices.data_ptr<int64_t>();
    const int num_threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "diff_cuda", ([&]
                                                                              {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
        auto stream = at::cuda::getCurrentCUDAStream();
        diff_cuda_kernel<scalar_t><<<(num_packs+num_threads-1)/num_threads, num_threads, 0, stream>>>(
            num_packs, num_feats, feat_dim, 
            feats_in.data_ptr<scalar_t>(), 
            feats_out.data_ptr<scalar_t>(), 
            pack_indices_ptr); }));
  }

  void inclusive_sum_cuda_impl(
      int64_t num,
      at::Tensor info,
      at::Tensor inclusive_sum)
  {

    int *info_ptr = info.data_ptr<int>();
    int *inclusive_sum_ptr = inclusive_sum.data_ptr<int>();

    void *temp_storage_ptr = NULL;
    uint64_t temp_storage_bytes = get_cub_storage_bytes(
        temp_storage_ptr, reinterpret_cast<uint *>(info_ptr), reinterpret_cast<uint *>(inclusive_sum_ptr), num);
    at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, device(at::DeviceType::CUDA).dtype(at::kByte));
    temp_storage_ptr = (void *)temp_storage.data_ptr<uint8_t>();

    CubDebugExit(cub::DeviceScan::InclusiveSum(temp_storage_ptr, temp_storage_bytes, info_ptr, inclusive_sum_ptr, num));
  }

  int sum_reduce_cuda_impl(
      int64_t num_feats,
      int64_t feat_dim,
      at::Tensor feats_in,
      at::Tensor feats_out,
      at::Tensor inclusive_sum)
  {

    int *inclusive_sum_ptr = inclusive_sum.data_ptr<int>();
    int cnt;

    const int num_threads = 1024;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "sum_reduce_cuda", ([&]
                                                                                    {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
      auto stream = at::cuda::getCurrentCUDAStream();
      cudaMemcpyAsync(&cnt, inclusive_sum_ptr + num_feats - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
      sum_reduce_cuda_kernel<scalar_t><<<(num_feats+num_threads-1)/num_threads, num_threads, 0, stream>>>(
          num_feats, feat_dim, 
          feats_in.data_ptr<scalar_t>(), 
          feats_out.data_ptr<scalar_t>(), 
          inclusive_sum_ptr); }));
    return cnt;
  }

  void cumsum_cuda_impl(
      int64_t num_feats,
      int64_t feat_dim,
      at::Tensor feats_in,
      at::Tensor feats_out,
      at::Tensor pack_indices,
      bool exclusive,
      bool reverse)
  {

    int64_t num_packs = pack_indices.size(0);
    int *pack_indices_ptr = pack_indices.data_ptr<int>();

    int offset = exclusive ? 1 : 0;

    const int num_threads = 256;
    if (reverse)
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "cumsum_cuda", ([&]
                                                                                  {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumsum_reverse_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset); }));
    }
    else
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "cumsum_cuda", ([&]
                                                                                  {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumsum_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset); }));
    }
  }

  void cumprod_cuda_impl(
      int64_t num_feats,
      int64_t feat_dim,
      at::Tensor feats_in,
      at::Tensor feats_out,
      at::Tensor pack_indices,
      bool exclusive,
      bool reverse)
  {

    int64_t num_packs = pack_indices.size(0);
    int *pack_indices_ptr = pack_indices.data_ptr<int>();

    int offset = exclusive ? 1 : 0;

    const int num_threads = 256;
    if (reverse)
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "cumprod_reverse_cuda", ([&]
                                                                                           {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumprod_reverse_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset); }));
    }
    else
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type(), "cumprod_cuda", ([&]
                                                                                   {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumprod_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset); }));
    }
  }

  ////////// generate rays //////////////////////////////////////////////////////////////////////////

  __global__ void
  generate_rays_cuda_kernel(
      uint num,
      uint width,
      uint height,
      float4x4 tf,
      float3 *ray_o,
      float3 *ray_d)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      uint px = tidx % width;
      uint py = tidx / height;

      float4 a = mul4x4(make_float4(0.0f, 0.0f, 1.0f, 0.0f), tf);
      float4 b = mul4x4(make_float4(px, py, 0.0f, 1.0f), tf);
      // float3 org = make_float3(M.m[3][0], M.m[3][1], M.m[3][2]);

      ray_o[tidx] = make_float3(a.x, a.y, a.z);
      ray_d[tidx] = make_float3(b.x, b.y, b.z);
    }
  }

  void generate_primary_rays_cuda_impl(
      uint width,
      uint height,
      float4x4 &tf,
      float3 *ray_o,
      float3 *ray_d)
  {
    uint num = width * height;

    generate_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(num, width, height, tf, ray_o, ray_d);
  }

  ////////// generate shadow rays /////////

  __global__ void
  plane_intersect_rays_cuda_kernel(
      uint num,
      float3 *ray_o,
      float3 *ray_d,
      float3 *output,
      float4 plane,
      uint *info)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      float3 org = ray_o[tidx];
      float3 dir = ray_d[tidx];

      float a = org.x * plane.x + org.y * plane.y + org.z * plane.z + plane.w;
      float b = dir.x * plane.x + dir.y * plane.y + dir.z * plane.z;

      if (fabs(b) > 1e-3)
      {
        float t = -a / b;
        if (t > 0.0f)
        {
          output[tidx] = make_float3(org.x + t * dir.x, org.y + t * dir.y, org.z + t * dir.z);
          info[tidx] = 1;
        }
        else
        {
          info[tidx] = 0;
        }
      }
      else
      {
        info[tidx] = 0;
      }
    }
  }

  __global__ void
  compactify_shadow_rays_cuda_kernel(
      uint num,
      float3 *p_in,
      float3 *p_out,
      uint *map,
      uint *info,
      uint *prefix_sum)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
    {
      p_out[prefix_sum[tidx]] = p_in[tidx];
      map[prefix_sum[tidx]] = tidx;
    }
  }

  __global__ void
  set_shadow_rays_cuda_kernel(
      uint num,
      float3 *src,
      float3 *dst,
      float3 light)
  {

    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      dst[tidx] = normalize(src[tidx] - light);
      src[tidx] = light;
    }
  }

  uint generate_shadow_rays_cuda_impl(
      uint num,
      float3 *ray_o,
      float3 *ray_d,
      float3 *src,
      float3 *dst,
      uint *map,
      float3 &light,
      float4 &plane,
      uint *info,
      uint *prefix_sum)
  {

    // set up memory for DeviceScan calls
    void *temp_storage_ptr = NULL;
    uint64_t temp_storage_bytes = get_cub_storage_bytes(temp_storage_ptr, info, prefix_sum, num);
    at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, device(at::DeviceType::CUDA).dtype(at::kByte));
    temp_storage_ptr = (void *)temp_storage.data_ptr<uint8_t>();

    uint cnt = 0;
    plane_intersect_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
        num, ray_o, ray_d, dst, plane, info);
    CubDebugExit(cub::DeviceScan::ExclusiveSum(
        temp_storage_ptr, temp_storage_bytes, info, prefix_sum, num));
    cudaMemcpy(&cnt, prefix_sum + num - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    compactify_shadow_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
        num, dst, src, map, info, prefix_sum);
    set_shadow_rays_cuda_kernel<<<(cnt + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(cnt, src, dst, light);

    return cnt;
  }

} // namespace kaolin
