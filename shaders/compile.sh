#!/usr/bin/sh
slangc compute.slang -profile spirv_1_6 -o compute.spv -O3 
slangc gemm.slang -profile spirv_1_6 -o gemm.spv -O3 
slangc gemm_coopmat.slang -profile spirv_1_6 -capability spvCooperativeMatrixKHR -o gemm_coopmat.spv -O3 
slangc gemm_coopmat_tiled.slang -profile spirv_1_6 -capability spvCooperativeMatrixKHR -capability spvGroupNonUniform -o gemm_coopmat_tiled.spv -O3 