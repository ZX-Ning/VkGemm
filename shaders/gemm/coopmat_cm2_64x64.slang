/**
 * A minimal VK_NV_cooperative_matrix2 GEMM for comparison with the tiled CM1
 * kernels. One 128-thread workgroup owns one flexible 64x64 matrix.
 *
 * Requirements:
 *   - M, N and K are multiples of 64.
 *   - The device supports a workgroup-scope 64x64x64 fp16/fp32 cooperative
 *     matrix with 128 workgroup invocations.
 *
 * Unlike the CM1 kernels, CM2 tensor addressing loads A, B and C directly
 * from their column-major storage. No explicit shared-memory staging is
 * needed in the shader.
 */

#define TILE_SIZE      64
#define WORKGROUP_SIZE 256

[[vk::constant_id(0)]]
const uint NUM_THREADS_X;

[[vk::constant_id(1)]]
const uint NUM_THREADS_Y;

[[vk::constant_id(2)]]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<half> mat_b;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float> mat_acc;

struct Shape {
    uint M;
    uint N;
    uint K;
};

[[vk_push_constant]]
ConstantBuffer<Shape> shape;

typealias MatA = linalg::CoopMat<
    half,
    MemoryScope.Workgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixA>;
typealias MatB = linalg::CoopMat<
    half,
    MemoryScope.Workgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixB>;
typealias MatACC = linalg::CoopMat<
    float,
    MemoryScope.Workgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixAccumulator>;

[[ForceInline]]
inline linalg::TensorLayout<2, linalg::CoopMatClampMode.Undefined>
make_column_major_layout(uint rows, uint cols) {
    linalg::TensorLayout<
        2,
        linalg::CoopMatClampMode.Undefined> layout;
    return layout.Dimension(rows, cols).Stride(1, rows);
}

[[shader("compute")]]
[[numthreads(WORKGROUP_SIZE, 1, 1)]]
void main(uint3 group_id: SV_GroupID) {
    const uint row_offset = group_id.x * TILE_SIZE;
    const uint col_offset = group_id.y * TILE_SIZE;

    let c_layout =
        make_column_major_layout(shape.M, shape.N)
            .Slice(
                row_offset,
                TILE_SIZE,
                col_offset,
                TILE_SIZE
            );
    var acc = MatACC.Load(mat_acc, 0, c_layout);

    for (uint k_offset = 0;
         k_offset < shape.K;
         k_offset += TILE_SIZE) {
        let a_layout =
            make_column_major_layout(shape.M, shape.K)
                .Slice(
                    row_offset,
                    TILE_SIZE,
                    k_offset,
                    TILE_SIZE
                );
        let b_layout =
            make_column_major_layout(shape.K, shape.N)
                .Slice(
                    k_offset,
                    TILE_SIZE,
                    col_offset,
                    TILE_SIZE
                );

        const MatA a = MatA.Load(mat_a, 0, a_layout);
        const MatB b = MatB.Load(mat_b, 0, b_layout);
        acc = linalg::coopMatMulAdd<float, false>(a, b, acc);
    }

    acc.Store(mat_acc, 0, c_layout);
}

