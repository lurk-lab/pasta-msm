// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/pasta.hpp>

typedef jacobian_t<vesta_t> point_t;
typedef xyzz_t<vesta_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef pallas_t scalar_t;

#include <msm/pippenger.cuh>
#include <spmvm/spmvm.cuh>

#ifndef __CUDA_ARCH__
extern "C" RustError cuda_pippenger_vesta(point_t *out, const affine_t points[], size_t npoints,
                                          const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars);
}

extern "C" RustError spmvm_vesta(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    return spmvm<scalar_t>(out, csr, scalars);
}

extern "C" RustError spmvm_cpu_vesta(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    return spmvm_cpu<scalar_t>(out, csr, scalars);
}
#endif
