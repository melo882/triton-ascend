# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import math
import random
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils, check_ub_mem_overflow, get_dtype_size

dtype_max_size_1d = {
    'int8 -> bfloat16': 32768,
    'uint8 -> bfloat16': 32768,
    'int8 -> float16': 65536,
    'uint8 -> float16': 65536,
    'int8 -> bool': 49056,
    'uint8 -> bool': 49056,
    'int8 -> float32': 32768,
    'uint8 -> float32': 32768,
    'int8 -> int32': 32768,
    'int8 -> uint32': 32768,
    'uint8 -> int32': 32768,
    'uint8 -> uint32': 32768,
    'uint8 -> int16': 32768,
    'uint8 -> uint16': 32768,
    'int8 -> int16': 32768,
    'int8 -> uint16': 84650,
    'int8 -> uint8': 196608,
    'int8 -> int8': 196608,
    'uint8 -> uint8': 196608,
    'uint8 -> int8': 196608,
    'int8 -> int64': 16384,
    'int8 -> uint64': 28206,
    'uint8 -> int64': 16384,
    'uint8 -> uint64': 16384,
    'bool -> int8': 49056,
    'int16 -> int8': 24576,
    'int16 -> uint8': 24576,
    'int16 -> float16': 98304,
    'int16 -> bfloat16': 32768,
    'int16 -> float32': 32768,
    'int16 -> int64': 16384,
    'int16 -> int16': 98304,
    'int16 -> int32': 32768,
    'int32 -> int8': 14016,
    'int32 -> uint8': 14016,
    'int32 -> int16': 32736,
    'int32 -> float16': 32768,
    'int32 -> bfloat16': 32768,
    'int32 -> float32': 49152,
    'int32 -> int64': 16384,
    'int32 -> int32': 49152,
    'float16 -> int8': 12288,
    'bfloat16 -> int8': 12288,
    'float16 -> uint8': 12288,
    'bfloat16 -> uint8': 12288,
    'float16 -> bfloat16': 98304,
    'float16 -> float16': 98304,
    'bfloat16 -> bfloat16': 98304,
    'bfloat16 -> float16': 98304,
    'float16 -> float32': 32768,
    'bfloat16 -> float32': 32768,
    'float16 -> int16': 32736,
    'bfloat16 -> int16': 32736,
    'float16 -> int32': 32768,
    'bfloat16 -> int32': 32768,
    'float16 -> int64': 16384,
    'bfloat16 -> int64': 16384,
    'float32 -> int8': 12288,
    'float32 -> uint8': 12288,
    'float32 -> int16': 32736,
    'float32 -> float16': 32768,
    'float32 -> bfloat16': 32768,
    'float32 -> int32': 49152,
    'float32 -> int64': 16384,
    'float32 -> float32': 49152,
    'int64 -> int8': 8928,
    'int64 -> uint8': 8928,
    'int64 -> int16': 14016,
    'int64 -> float16': 24576,
    'int64 -> bfloat16': 24576,
    'int64 -> int32': 24576,
    'int64 -> int64': 24576,
    'int64 -> bool': 28206,
    #uint64
    'uint64 -> int8': 8928,
    'uint64 -> int16': 14016,
    'uint64 -> int32': 24576,
    'uint64 -> int64': 24576,
    'uint64 -> float16': 24576,
    'uint64 -> bfloat16': 24576,
    'uint64 -> uint8': 8928,
    'uint64 -> uint16': 14016,
    'uint64 -> uint32': 24576,
    'uint64 -> uint64': 24576,
    'uint64 -> bool': 28206,
    # uint16
    'uint16 -> int8': 24576,
    'uint16 -> uint8': 24576,
    'uint16 -> float16': 98304,
    'uint16 -> bfloat16': 32768,
    'uint16 -> float32': 32768,
    'uint16 -> int64': 16384,
    'uint16 -> int16': 98304,
    'uint16 -> uint16': 98304,
    'uint16 -> uint32': 32768,
    'uint16 -> uint64': 16384,
    'uint16 -> bool': 49056,
    #uint32
    'uint32 -> int8': 14016,
    'uint32 -> uint8': 14016,
    'uint32 -> int16': 32736,
    'uint32 -> float16': 32768,
    'uint32 -> bfloat16': 32768,
    'uint32 -> float32': 49152,
    'uint32 -> int64': 16384,
    'uint32 -> int32': 49152,
    'uint32 -> uint32': 49152,
    'uint32 -> uint16': 32736,
    'uint32 -> uint64': 16384,
    'uint32 -> bool': 24480,
}


@triton.jit
def cast_to_bool(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int1)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_uint8(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                  DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint8)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_uint16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                   DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint16)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_uint32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                   DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint32)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_uint64(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                   DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint64)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i8(output_ptr, x_ptr, x_stride, y_stride, z_stride,
               DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int8)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int16)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int32)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i64(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int64)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_fp32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.float32)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_fp16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.float16)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_bf16(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                 DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.bfloat16)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_uint32(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                   DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.uint32)
    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_int64(output_ptr, x_ptr, x_stride, y_stride, z_stride,
                  DIM: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    if DIM == 1:
        xidx = tl.arange(0, XB)
        idx = xidx * x_stride
    elif DIM == 2:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        idx = xidx[:, None] * x_stride + yidx[None, :] * y_stride
    elif DIM == 3:
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = xidx[:, None, None] * x_stride + yidx[None, :, None] * y_stride + zidx[None, None, :] * z_stride

    X = tl.load(x_ptr + idx)
    ret = tl.cast(X, dtype=tl.int64)
    tl.store(output_ptr + idx, ret)


triton_func_map = {
    "bool": cast_to_bool,
    "int8": cast_to_i8,
    "int16": cast_to_i16,
    "int32": cast_to_i32,
    "float16": cast_to_fp16,
    "bfloat16": cast_to_bf16,
    "float32": cast_to_fp32,
    "uint32": cast_to_uint32,
    "int64": cast_to_int64,
    "uint8": cast_to_uint8,
    "uint16": cast_to_uint16,
    "uint64": cast_to_uint64
}


def structParam(x0):
    dim = x0.dim()
    stride0, stride1, stride2 = 0, 0, 0
    shape0, shape1, shape2 = 0, 0, 0
    if dim >= 1:
        stride0 = x0.stride(0)
        shape0 = x0.shape[0]
    if dim >= 2:
        stride1 = x0.stride(1)
        shape1 = x0.shape[1]
    if dim == 3:
        stride2 = x0.stride(2)
        shape2 = x0.shape[2]
    return dim, stride0, stride1, stride2, shape0, shape1, shape2


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('srcDtype',
                         ['int8', 'int16', 'int32', 'int64', 'bool', 'uint16', 'uint32', 'uint64']
                         )
@pytest.mark.parametrize('dstDtype',
                         ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool',
                          'uint16', 'uint32', 'uint64']
                         )
def test_cast_int(srcDtype, dstDtype, shape):
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)
    dtype_size = max(srcBytes, dstBytes)
    if 'int8' in {srcDtype, dstDtype} or 'uint8' in {srcDtype, dstDtype}:
        key = f"{srcDtype} -> {dstDtype}"
        max_size = dtype_max_size_1d[key]
        numel = math.prod(shape)
        if len(shape) == 1:  # 1-D
            if numel > max_size:
                pytest.skip(f"1-D shape {shape} numel={numel} > max_size={max_size}")
        else:  # N-D
            if (shape[-1] * dtype_size) % 32 == 0:  # 尾轴 32 B 对齐
                limit = max_size // 8
            else:
                limit = max_size // 8 // 32
            if numel > limit:
                pytest.skip(f"ND shape {shape} numel={numel} > limit={limit}")
    else:
        if dtype_size * math.prod(shape) >= TestUtils.ub_size / 12:
            pytest.skip(f"UB memory estimate overflow")

    x0 = test_common.generate_tensor(shape, srcDtype)
    torch_res = x0.to(eval("torch." + dstDtype))
    x0 = x0.npu()
    triton_func = triton_func_map.get(dstDtype, None)
    assert triton_func is not None, f"triton_func not Found, srcDtype:{srcDtype}, dstDtype:{dstDtype}"
    triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()
    dim, stride0, stride1, stride2, XB, YB, ZB = structParam(x0)
    assert 0 <= dim <= 3, f"dim out of range [0, 3], dim:{dim}"
    triton_func[1, 1, 1](triton_res, x0, stride0, stride1, stride2, dim, XB, YB, ZB)
    test_common.validate_cmp(dstDtype, triton_res, torch_res)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('srcDtype',
                         ['float16', 'float32', 'bfloat16', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool',
                          'uint16', 'uint32', 'uint64']
                         )
@pytest.mark.parametrize('dstDtype',
                         ['float16', 'float32', 'bfloat16']
                         )
def test_cast_float(srcDtype, dstDtype, shape):
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)
    dtype_size = max(srcBytes, dstBytes)
    if 'int8' in {srcDtype, dstDtype} or 'uint8' in {srcDtype, dstDtype}:
        key = f"{srcDtype} -> {dstDtype}"
        max_size = dtype_max_size_1d[key]
        numel = math.prod(shape)
        if len(shape) == 1:  # 1-D
            if numel > max_size:
                pytest.skip(f"1-D shape {shape} numel={numel} > max_size={max_size}")
        else:  # N-D
            if (shape[-1] * dtype_size) % 32 == 0:  # 尾轴 32 B 对齐
                limit = max_size // 8
            else:
                limit = max_size // 8 // 32
            if numel > limit:
                pytest.skip(f"ND shape {shape} numel={numel} > limit={limit}")
    else:
        if dtype_size * math.prod(shape) >= TestUtils.ub_size / 12:
            pytest.skip(f"UB memory estimate overflow")

    x0 = test_common.generate_tensor(shape, srcDtype)
    torch_res = x0.to(eval("torch." + dstDtype))
    x0 = x0.npu()
    triton_func = triton_func_map.get(dstDtype, None)
    assert triton_func is not None, f"triton_func not Found, srcDtype:{srcDtype}, dstDtype:{dstDtype}"
    triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()
    dim, stride0, stride1, stride2, XB, YB, ZB = structParam(x0)
    assert 0 <= dim <= 3, f"dim out of range [0, 3], dim:{dim}"
    triton_func[1, 1, 1](triton_res, x0, stride0, stride1, stride2, dim, XB, YB, ZB)
    test_common.validate_cmp(dstDtype, triton_res, torch_res)


@triton.jit
def cast_to_nd(
        out_ptr, in_ptr,
        D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
        D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr,
):
    dtype = out_ptr.type.element_ty

    off = tl.arange(0, D1) * (D2 * D3 * D4 * D5 * D6 * D7 * D8)
    if (D2 * D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, None, ] + tl.arange(0, D2)[None, :] * (D3 * D4 * D5 * D6 * D7 * D8)
    if (D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, None] + tl.arange(0, D3)[None, None, :] * (D4 * D5 * D6 * D7 * D8)
    if (D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, None] + tl.arange(0, D4)[None, None, None, :] * (D5 * D6 * D7 * D8)
    if (D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, :, None] + tl.arange(0, D5)[None, None, None, None, :] * (D6 * D7 * D8)
    if (D6 * D7 * D8) > 1:
        off = off[:, :, :, :, :, None] + tl.arange(0, D6)[None, None, None, None, None, :] * (D7 * D8)
    if (D7 * D8) > 1:
        off = off[:, :, :, :, :, :, None] + tl.arange(0, D7)[None, None, None, None, None, None, :] * D8
    if D8 > 1:
        off = off[:, :, :, :, :, :, :, None] + tl.arange(0, D8)[None, None, None, None, None, None, None, :]

    mask = off < (D1 * D2 * D3 * D4 * D5 * D6 * D7 * D8)
    x = tl.load(in_ptr + off)
    y = tl.cast(x, dtype)
    tl.store(out_ptr + off, y)


@pytest.mark.parametrize('srcDtype',
                         ['float16', 'float32', 'bfloat16', 'uint8','uint16', 'uint32', 'uint64']
                         )
@pytest.mark.parametrize('dstDtype',
                         ['float16', 'float32', 'bfloat16', 'uint8','uint16', 'uint32', 'uint64']
                         )
@pytest.mark.parametrize('shape', TestUtils.full_shape_4_8d)
def test_cast_nd_float(srcDtype, dstDtype, shape):
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)
    dtype_size = max(srcBytes, dstBytes)
    if 'int8' in {srcDtype, dstDtype} or 'uint8' in {srcDtype, dstDtype}:
        key = f"{srcDtype} -> {dstDtype}"
        max_size = dtype_max_size_1d[key]
        numel = math.prod(shape)
        if len(shape) == 1:  # 1-D
            if numel > max_size:
                pytest.skip(f"1-D shape {shape} numel={numel} > max_size={max_size}")
        else:  # N-D
            if (shape[-1] * dtype_size) % 32 == 0:  # 尾轴 32 B 对齐
                limit = max_size // 8
            else:
                limit = max_size // 8 // 32
            if numel > limit:
                pytest.skip(f"ND shape {shape} numel={numel} > limit={limit}")
    else:
        if dtype_size * math.prod(shape) >= TestUtils.ub_size / 12:
            pytest.skip(f"UB memory estimate overflow")

    # x0 = test_common.generate_tensor(shape, srcDtype)
    x0 = test_common.generate_tensor_new(shape, srcDtype, extreme_ratio=0, special_ratio=0, precision_ratio=0,  seed=1000).npu()

    torch_res = x0.cpu().to(eval("torch." + dstDtype))
    print('input: ', x0.cpu())
    triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()

    triton_shape = [*shape]
    while len(triton_shape) < 8:
        triton_shape.append(1)
    grid = (1,)
    cast_to_nd[grid](triton_res, x0, *triton_shape)
    print('Triton result: ', triton_res.cpu())
    print('Torch result: ', torch_res.cpu())
    diff_mask = triton_res.cpu() != torch_res.cpu()          # 逐元素布尔张量

    if diff_mask.any():                          # 只要有不同
        idx = diff_mask.nonzero(as_tuple=False)  # N×ndim 的坐标矩阵
        print(f'共有 {idx.size(0)} 处不同')
    # 只打印前 10 条，防止刷屏
        for i, pos in enumerate(idx[:10]):
            pos = tuple(pos.tolist())            # 把 torch.Tensor 转 tuple
            print(f'pos={pos}  |  '
                  f'Triton={triton_res[pos].item()}  |  '
                  f'Torch={torch_res[pos].item()}')
    else:
        print('完全相等')
    test_common.validate_cmp(dstDtype, triton_res.cpu(), torch_res.cpu())


@pytest.mark.parametrize('srcDtype',
                         ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
                         )
@pytest.mark.parametrize('dstDtype',
                         ['int8', 'int16', 'int32', 'int64', 'uint8', 'float16', 'float32', 'bfloat16']
                         )
@pytest.mark.parametrize('shape', TestUtils.full_shape_4_8d)
def test_cast_nd_int(srcDtype, dstDtype, shape):
    print('shape: ', shape)
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)
    dtype_size = max(srcBytes, dstBytes)
    if 'int8' in {srcDtype, dstDtype} or 'uint8' in {srcDtype, dstDtype}:
        key = f"{srcDtype} -> {dstDtype}"
        max_size = dtype_max_size_1d[key]
        numel = math.prod(shape)
        if len(shape) == 1:  # 1-D
            if numel > max_size:
                pytest.skip(f"1-D shape {shape} numel={numel} > max_size={max_size}")
        else:  # N-D
            if (shape[-1] * dtype_size) % 32 == 0:  # 尾轴 32 B 对齐
                limit = max_size // 8
            else:
                limit = max_size // 8 // 32
            if numel > limit:
                pytest.skip(f"ND shape {shape} numel={numel} > limit={limit}")
    else:
        if dtype_size * math.prod(shape) >= TestUtils.ub_size / 12:
            pytest.skip(f"UB memory estimate overflow")

    x0 = test_common.generate_tensor(shape, srcDtype, extreme_ratio=0, special_ratio=0, seed=1000)
    torch_res = x0.cpu().to(eval("torch." + dstDtype))
    x0 = x0.npu()

    triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()

    triton_shape = [*shape]
    while len(triton_shape) < 8:
        triton_shape.append(1)
    grid = (1,)
    cast_to_nd[grid](triton_res, x0, *triton_shape)
    test_common.validate_cmp(dstDtype, triton_res, torch_res)


# ===========================================
@triton.jit
def cast_to_nd_satu(
        out_ptr, in_ptr,
        D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
        D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr,
):
    dtype = out_ptr.type.element_ty

    off = tl.arange(0, D1) * (D2 * D3 * D4 * D5 * D6 * D7 * D8)
    if (D2 * D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, None, ] + tl.arange(0, D2)[None, :] * (D3 * D4 * D5 * D6 * D7 * D8)
    if (D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, None] + tl.arange(0, D3)[None, None, :] * (D4 * D5 * D6 * D7 * D8)
    if (D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, None] + tl.arange(0, D4)[None, None, None, :] * (D5 * D6 * D7 * D8)
    if (D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, :, None] + tl.arange(0, D5)[None, None, None, None, :] * (D6 * D7 * D8)
    if (D6 * D7 * D8) > 1:
        off = off[:, :, :, :, :, None] + tl.arange(0, D6)[None, None, None, None, None, :] * (D7 * D8)
    if (D7 * D8) > 1:
        off = off[:, :, :, :, :, :, None] + tl.arange(0, D7)[None, None, None, None, None, None, :] * D8
    if D8 > 1:
        off = off[:, :, :, :, :, :, :, None] + tl.arange(0, D8)[None, None, None, None, None, None, None, :]

    mask = off < (D1 * D2 * D3 * D4 * D5 * D6 * D7 * D8)
    x = tl.load(in_ptr + off)
    y = extension.cast(x, dtype, overflow_mode='saturate')
    tl.store(out_ptr + off, y)


# TODO 涉及场景 包含参数
# 修改kernel 入参
@triton.jit
def cast_to_nd_with_parameter(
        out_ptr, in_ptr,
        D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
        D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr,
        fp_downcast_rounding: tl.constexpr, bitcast: tl.constexpr):
    dtype = out_ptr.type.element_ty

    off = tl.arange(0, D1) * (D2 * D3 * D4 * D5 * D6 * D7 * D8)
    if (D2 * D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, None, ] + tl.arange(0, D2)[None, :] * (D3 * D4 * D5 * D6 * D7 * D8)
    if (D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, None] + tl.arange(0, D3)[None, None, :] * (D4 * D5 * D6 * D7 * D8)
    if (D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, None] + tl.arange(0, D4)[None, None, None, :] * (D5 * D6 * D7 * D8)
    if (D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, :, None] + tl.arange(0, D5)[None, None, None, None, :] * (D6 * D7 * D8)
    if (D6 * D7 * D8) > 1:
        off = off[:, :, :, :, :, None] + tl.arange(0, D6)[None, None, None, None, None, :] * (D7 * D8)
    if (D7 * D8) > 1:
        off = off[:, :, :, :, :, :, None] + tl.arange(0, D7)[None, None, None, None, None, None, :] * D8
    if D8 > 1:
        off = off[:, :, :, :, :, :, :, None] + tl.arange(0, D8)[None, None, None, None, None, None, None, :]

    mask = off < (D1 * D2 * D3 * D4 * D5 * D6 * D7 * D8)
    x = tl.load(in_ptr + off)
    y = tl.cast(x, dtype, fp_downcast_rounding=fp_downcast_rounding, bitcast=bitcast)
    tl.store(out_ptr + off, y)


# 添加典型值
shapes_list = [
    (13,),
    (9,),
    (1,),
    (23,),
    (300,),
    (600,),

    (1, 23),
    (1, 1),
    (25, 5),
    (11, 33),
    (1000, 1),
    (20, 27),

    (3, 3, 3),
    (1, 1, 23),
    (1, 22, 39),
    (27, 1, 39),
    (27, 22, 1),
    (1, 1, 23),
    (23, 1, 1),
    (1, 23, 1),
    (37, 5, 3),
    (2, 29, 4),
    (7, 31, 7),
    (3, 5, 8),
    (7, 17, 15),
    (25, 5, 16),
    (13, 5, 31),
    (9, 11, 32),
    (7, 11, 33),
    (1, 1, 1),
    (1, 23, 1),
    (7, 17, 41),
    (1, 1000, 1),
    (13, 27, 10),
    (3, 27, 5),
    (9, 31, 25),
    (2, 25, 9),
    (5, 25, 3),
    (3, 15, 33),
    (2, 9, 15),
    (19, 7, 3),
    (1, 1, 1, 23),
    (1, 1, 1, 1),
    (7, 5, 3, 4),
    (4, 5, 32, 16),
    (7, 17, 5, 4),
    (1, 1000, 1, 1),
    (1, 1, 1000, 1),
    (1, 1, 1, 1, 23),
    (1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2),
    (2, 4, 5, 4, 3),
    (1, 1, 1, 1000, 1),
    (1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2),
    (2, 4, 2, 3, 2, 8),

    (1, 1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2, 2),
    (2, 2, 3, 2, 2, 2, 5),

    (1, 1, 1, 1, 1, 1, 1, 1),
    (2, 2, 2, 1, 2, 2, 2, 1),
]

# ++++++++++++


extreme_range_dict = {
    'int8': [-128, 127],
    'int16': [-32768, 32767],
    'int32': [-2147483648, 2147483647],
    'int64': [-9223372036854775808, 9223372036854775807],
    'uint8': [0, 255],
    'uint16': [0, 65535],
    'uint32': [0, 4294967295],
    'uint64': [0, 18446744073709551615],
    'float16': [-65504, 65504],
    'float32': [-3.4e+38, 3.4e+38],
    'bfloat16': [-3.38953e+38, 3.38953e+38],
    'bool': [-1, 1],  # 不生效，实际只在True和False中取值
    'fp8e4m3': [-240, 240],
    'fp8e5m2': [-57344, 57344],
    'fp8e5b16': [-57344, 57344],
    'fp4': [-6, 6],
}

# TODO fp_downcast_rounding参数
transfer_list = [
    ['float32', 'float16'],
    ['float32', 'bfloat16'],
    ['float32', 'float32'],
    ['float16', 'float16'],
    ['bfloat16', 'bfloat16'],
    ['float32', 'fp8e4m3'],
    ['float32', 'fp8e5m2'],
    ['float16', 'fp8e4m3'],
    ['float16', 'fp8e4m3'],
    ['bfloat16', 'fp8e5m2'],
    ['bfloat16', 'fp8e4m3'],
    ['fp8e5m2', 'fp8e5m2'],
    ['fp8e4m3', 'fp8e4m3'],
]


def cast_with_rounding(x: torch.Tensor, dtype_str: str, rounding: str):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp8e5m2": torch.float8_e5m2,
        "fp8e4m3": torch.float8_e4m3fn,
    }
    target_dtype = dtype_map[dtype_str]

    # 如果目标类型和源类型相同，则不做任何操作
    if x.dtype == target_dtype:
        return x

    # 升精度到 float32 再做降精计算
    x = x.to(torch.float32)

    if rounding == "rtne":
        return x.to(target_dtype)

    elif rounding == "rtz":
        if target_dtype == torch.float16:
            scale = 2 ** 10
        elif target_dtype == torch.bfloat16:
            scale = 2 ** 7
        else:
            return x

        truncated = torch.where(x > 0, torch.floor(x * scale) / scale,
                                torch.ceil(x * scale) / scale)
        return truncated.to(target_dtype)


@pytest.mark.parametrize('fp_downcast_rounding', ["rtne", "rtz"])
@pytest.mark.parametrize('sigtype, dstDtype', transfer_list)
@pytest.mark.parametrize('shape', shapes_list)
def test_cast_fp_downcast_rounding(fp_downcast_rounding, dstDtype, sigtype, shape, bitcast=False):
    if sigtype == 'fp8e4m3':
        sigtype_new = 'float8_e4m3fn'
        srcBytes = get_dtype_size(sigtype_new)
    elif sigtype == 'fp8e5m2':
        sigtype_new = 'float8_e5m2'
        srcBytes = get_dtype_size(sigtype_new)
    else:
        srcBytes = get_dtype_size(sigtype)

    if dstDtype == 'fp8e4m3':
        dstDtype_new = 'float8_e4m3fn'
        dstBytes = get_dtype_size(dstDtype_new)
    elif dstDtype == 'fp8e5m2':
        dstDtype_new = 'float8_e5m2'
        dstBytes = get_dtype_size(dstDtype_new)
    else:
        dstBytes = get_dtype_size(dstDtype)

    dtype_size = max(srcBytes, dstBytes)


    if dstDtype == 'int8':
        if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 100):
            pytest.skip(f"UB memory estimate overflow")
            return
    elif dtype_size * math.prod(shape) >= (TestUtils.ub_size / 12):
        pytest.skip(f"UB memory estimate overflow")

    x0 = test_common.generate_tensor(shape, sigtype)
    if dstDtype == 'fp8e4m3':
        dstDtype_new = 'float8_e4m3fn'
        torch_res = cast_with_rounding(x0, sigtype, fp_downcast_rounding).to(eval("torch." + dstDtype_new))
        triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype_new)).npu()
    elif dstDtype == 'fp8e5m2':
        dstDtype_new = 'float8_e5m2'
        torch_res = cast_with_rounding(x0, sigtype, fp_downcast_rounding).to(eval("torch." + dstDtype_new))
        triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype_new)).npu()
    else:
        torch_res = cast_with_rounding(x0, sigtype, fp_downcast_rounding).to(eval("torch." + dstDtype))
        triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()

    x0 = x0.npu()

    # triton_res = torch.empty(shape, dtype=eval("torch." + dstDtype)).npu()

    triton_shape = [*shape]
    while len(triton_shape) < 8:
        triton_shape.append(1)
    grid = (1,)
    cast_to_nd_with_parameter[grid](triton_res, x0, *triton_shape, fp_downcast_rounding, bitcast)
    test_common.validate_cmp(dstDtype, triton_res, torch_res)


# set bitcast=True
def bitcast_reference(x, src_dtype, dst_dtype):
    src_t = x.to(eval("torch." + dst_dtype))

    src_bytes = torch.empty([], dtype=eval("torch." + src_dtype)).element_size()
    dst_bytes = torch.empty([], dtype=eval("torch." + dst_dtype)).element_size()

    if src_bytes != dst_bytes:
        raise ValueError(
            f"bitcast requires same element size, but got {src_bytes} vs {dst_bytes}"
        )

    return src_t.view(eval("torch." + dst_dtype))


@pytest.mark.parametrize('srcDtype',
                         ['int8', 'float16', 'float32', 'int16', 'int32', 'int64', 'bfloat16', 'int64', 'uint8',
                          'uint16', 'uint32', 'uint64', 'fp8e5m2', 'fp8e4m3'])
@pytest.mark.parametrize('dstDtype',
                         ['int8', 'float16', 'float32', 'int16', 'int32', 'int64', 'bfloat16', 'int64', 'uint8',
                          'uint16', 'uint32', 'uint64', 'fp8e5m2', 'fp8e4m3'])
@pytest.mark.parametrize('shape', shapes_list)
def test_cast_bitcast_on(srcDtype, dstDtype, shape, bitcast=True, fp_downcast_rounding="rtne"):
    srcBytes = get_dtype_size(srcDtype)
    dstBytes = get_dtype_size(dstDtype)

    dtype_size = max(srcBytes, dstBytes)
    if dstDtype == 'int8':
        if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 100):
            pytest.skip(f"UB memory estimate overflow")
            return
    elif dtype_size * math.prod(shape) >= (TestUtils.ub_size / 12):
        pytest.skip(f"UB memory estimate overflow")

    if srcBytes != dstBytes:
        pytest.skip(f"bitcast invalid: size mismatch {srcBytes} vs {dstBytes}")
    elif srcBytes == 'fp8e5m2' and dstDtype == 'fp8e4m3':
        pytest.skip(f"fp8e5m2 cannot bitcast to fp8e4m3")
    elif srcBytes == 'fp8e4m3' and dstDtype == 'fp8e5m2':
        pytest.skip(f"fp8e4m3 cannot bitcast to fp8e5m2")

    x0 = test_common.generate_tensor(shape, srcDtype)
    x0_npu = x0.npu()
    dtype_new = test_common.get_torch_typename(dstDtype)
    torch_res = x0.view(dtype_new)
    triton_res = torch.empty(shape, dtype=dtype_new).npu()

    triton_shape = [*shape] + [1] * (8 - len(shape))
    cast_to_nd_with_parameter[(1,)](
        triton_res, x0_npu, *triton_shape, fp_downcast_rounding, bitcast
    )

    assert triton_res.dtype == dtype_new

    test_common.validate_cmp(dstDtype, triton_res, torch_res)


# overflow_mode
transfer_list = [
    ('int64', 'int32'),
    ('int64', 'uint32'),
    ('uint64', 'uint32'),
    ('uint64', 'int32'),
    ('int64', 'int16'),
    ('int64', 'uint16'),
    ('uint64', 'int16'),
    ('uint64', 'uint16'),
    ('int64', 'int8'),
    ('int64', 'uint8'),
    ('uint64', 'int8'),
    ('uint64', 'uint8'),
    ('int32', 'int16'),
    ('int32', 'int8'),
    ('int32', 'uint8'),
    ('int32', 'uint16'),
    ('int32', 'uint8'),
    ('uint32', 'int16'),
    ('uint32', 'int8'),
    ('uint32', 'uint16'),
    ('uint32', 'uint8'),
    ('int16', 'int8'),
    ('int16', 'uint8'),
    ('uint16', 'int8'),
    ('uint16', 'uint8'),
]


# ('int64', 'uint32'),
# ('int64', 'uint16'),
# ('int32', 'uint16'),
# ('uint64', 'uint32'),
# ('uint64', 'uint16'),
# ('uint32', 'uint16'),
import numpy as np


def saturate_cast_numpy(x, dst_dtype):
    dtype_map = {
        torch.uint8: np.uint8, torch.uint16: np.uint16,
        torch.uint32: np.uint32, torch.uint64: np.uint64,
        torch.int8: np.int8, torch.int16: np.int16,
        torch.int32: np.int32, torch.int64: np.int64,
    }

    np_dtype = dtype_map[dst_dtype]
    info = np.iinfo(np_dtype)

    np_x = x.cpu().numpy()

    # 关键修复：确保 min/max 是 numpy scalar 且类型正确
    min_val = np.dtype(np_dtype).type(info.min)
    max_val = np.dtype(np_dtype).type(info.max)

    np_x = np.clip(np_x, min_val, max_val).astype(np_dtype)
    return torch.from_numpy(np_x).to(x.device)


@triton.jit
def cast_to_nd_with_overflow(
        out_ptr, in_ptr,
        D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
        D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr,
        overflow_mode: tl.constexpr):
    dtype = out_ptr.type.element_ty

    off = tl.arange(0, D1) * (D2 * D3 * D4 * D5 * D6 * D7 * D8)
    if (D2 * D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, None, ] + tl.arange(0, D2)[None, :] * (D3 * D4 * D5 * D6 * D7 * D8)
    if (D3 * D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, None] + tl.arange(0, D3)[None, None, :] * (D4 * D5 * D6 * D7 * D8)
    if (D4 * D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, None] + tl.arange(0, D4)[None, None, None, :] * (D5 * D6 * D7 * D8)
    if (D5 * D6 * D7 * D8) > 1:
        off = off[:, :, :, :, None] + tl.arange(0, D5)[None, None, None, None, :] * (D6 * D7 * D8)
    if (D6 * D7 * D8) > 1:
        off = off[:, :, :, :, :, None] + tl.arange(0, D6)[None, None, None, None, None, :] * (D7 * D8)
    if (D7 * D8) > 1:
        off = off[:, :, :, :, :, :, None] + tl.arange(0, D7)[None, None, None, None, None, None, :] * D8
    if D8 > 1:
        off = off[:, :, :, :, :, :, :, None] + tl.arange(0, D8)[None, None, None, None, None, None, None, :]

    x = tl.load(in_ptr + off)
    y = extension.cast(x, dtype, overflow_mode=overflow_mode)
    tl.store(out_ptr + off, y)


if __name__ == "__main__":
    for shape in [(3,), (3, 3), (3, 3, 3)]:
        for srcDtype in ['int8', 'float32', 'bool']:
            for dstDtype in ['int8', 'float32', 'bool']:
                test_cast(srcDtype, dstDtype, shape)
