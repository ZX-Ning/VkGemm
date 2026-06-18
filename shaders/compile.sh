#!/usr/bin/sh
slangc compute.slang -profile spirv_1_6 -o compute.spv -O3 
slangc gemm.slang -profile spirv_1_6 -o gemm.spv -O3 