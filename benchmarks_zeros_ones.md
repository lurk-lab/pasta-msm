# Benchmarks

## Table of Contents

- [Benchmark Results](#benchmark-results)
    - [GPU](#gpu)

## Benchmark Results

### GPU

|                            | `lurkrs`                  | `lurkrs no zeros and ones`           |
|:---------------------------|:--------------------------|:------------------------------------ |
| **`#1, 7941351 scalars`**  | `546.36 ms` (✅ **1.00x**) | `562.64 ms` (✅ **1.03x slower**)     |
| **`#2, 9699051 scalars`**  | `148.42 ms` (✅ **1.00x**) | `174.88 ms` (❌ *1.18x slower*)       |
| **`#5, 7941351 scalars`**  | `554.75 ms` (✅ **1.00x**) | `567.34 ms` (✅ **1.02x slower**)     |
| **`#6, 9699051 scalars`**  | `323.56 ms` (✅ **1.00x**) | `355.95 ms` (✅ **1.10x slower**)     |
| **`#9, 7941351 scalars`**  | `556.20 ms` (✅ **1.00x**) | `570.82 ms` (✅ **1.03x slower**)     |
| **`#10, 9699051 scalars`** | `348.17 ms` (✅ **1.00x**) | `391.35 ms` (❌ *1.12x slower*)       |
| **`#13, 7941351 scalars`** | `556.44 ms` (✅ **1.00x**) | `570.68 ms` (✅ **1.03x slower**)     |
| **`#14, 9699051 scalars`** | `466.73 ms` (✅ **1.00x**) | `503.20 ms` (✅ **1.08x slower**)     |
| **`#17, 7941351 scalars`** | `558.69 ms` (✅ **1.00x**) | `571.33 ms` (✅ **1.02x slower**)     |
| **`#18, 9699051 scalars`** | `564.64 ms` (✅ **1.00x**) | `597.10 ms` (✅ **1.06x slower**)     |
| **`#21, 7941351 scalars`** | `558.72 ms` (✅ **1.00x**) | `574.41 ms` (✅ **1.03x slower**)     |
| **`#22, 9699051 scalars`** | `619.22 ms` (✅ **1.00x**) | `646.57 ms` (✅ **1.04x slower**)     |
| **`#25, 7941351 scalars`** | `559.44 ms` (✅ **1.00x**) | `574.14 ms` (✅ **1.03x slower**)     |
| **`#26, 9699051 scalars`** | `680.43 ms` (✅ **1.00x**) | `710.99 ms` (✅ **1.04x slower**)     |
| **`#29, 7941351 scalars`** | `558.92 ms` (✅ **1.00x**) | `571.54 ms` (✅ **1.02x slower**)     |
| **`#30, 9699051 scalars`** | `704.44 ms` (✅ **1.00x**) | `734.09 ms` (✅ **1.04x slower**)     |

---
Made with [criterion-table](https://github.com/nu11ptr/criterion-table)

