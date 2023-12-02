// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/pasta.hpp>

typedef jacobian_t<pallas_t> point_t;
typedef xyzz_t<pallas_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef vesta_t scalar_t;

#include <msm/pippenger.cuh>
#include <spmvm/spmvm.cuh>

#ifndef __CUDA_ARCH__

extern "C" void add_test_pallas(void)
{
    add_test();
}

// extern "C" RustError scalar_add_test_pallas(size_t n, const scalar_t *x, const scalar_t *y, scalar_t *out)
// {
//     return scalar_add_test<scalar_t>(n, x, y, out);
// }

extern "C" RustError cuda_pippenger_pallas(point_t *out, const affine_t points[], size_t npoints,
                                           const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars);
}

extern "C" RustError spmvm_pallas(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    return spmvm<scalar_t>(out, csr, scalars);
}

extern "C" RustError spmvm_cpu_pallas(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    return spmvm_cpu<scalar_t>(out, csr, scalars);
}
#endif
