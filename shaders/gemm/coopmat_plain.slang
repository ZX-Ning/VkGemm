#define TILE_SIZE 16
// #define SUBGROUP_SIZE 32

[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<half> mat_b;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float> mat_acc;

struct Shape {
    /**
     * A: M X K; B: K X N; ACC: M X N
     * ALL COL-Major
     */
    uint M;
    uint N;
    uint K;
};

[[vk_push_constant]]
ConstantBuffer<Shape> shape;

typealias MatA = linalg::CoopMat<
    half,
    MemoryScope.Subgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixA>;
typealias MatB = linalg::CoopMat<
    half,
    MemoryScope.Subgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixB>;
typealias MatACC = linalg::CoopMat<
    float,
    MemoryScope.Subgroup,
    TILE_SIZE,
    TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixAccumulator>;

const static let COL_MAJ = linalg::CoopMatMatrixLayout::ColumnMajor;

[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID) {
    uint row_offset = gid.x * TILE_SIZE;
    uint col_offset = gid.y * TILE_SIZE;

    MatACC acc = MatACC.Load<COL_MAJ>(
        mat_acc, col_offset * shape.M + row_offset, shape.M
    );

    for (uint t = 0; t < shape.K; t += TILE_SIZE) {
        // A: fix x=row_offset, slide y=t
        // B: fix y=col_offset, slide x=t
        MatA a = MatA.Load<COL_MAJ>(mat_a, t * shape.M + row_offset, shape.M);
        MatB b = MatB.Load<COL_MAJ>(mat_b, col_offset * shape.K + t, shape.K);
        acc = linalg::coopMatMulAdd<float, false>(a, b, acc);
    }

    acc.Store<COL_MAJ>(mat_acc, col_offset * shape.M + row_offset, shape.M);
}

