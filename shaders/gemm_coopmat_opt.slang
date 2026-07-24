#define TILE_SIZE      32
#define COOP_TILE_SIZE 16

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
    COOP_TILE_SIZE,
    COOP_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixA>;
typealias MatB = linalg::CoopMat<
    half,
    MemoryScope.Subgroup,
    COOP_TILE_SIZE,
    COOP_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixB>;
typealias MatACC = linalg::CoopMat<
    float,
    MemoryScope.Subgroup,
    COOP_TILE_SIZE,
    COOP_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixAccumulator>;

const static let COL_MAJ = linalg::CoopMatMatrixLayout::ColumnMajor;

[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID) {
    uint row_offset = gid.x * TILE_SIZE;
    uint col_offset = gid.y * TILE_SIZE;

    MatACC acc00 = MatACC.Load<COL_MAJ>(mat_acc, col_offset * shape.M + row_offset, shape.M);
    MatACC acc01 = MatACC.Load<COL_MAJ>(mat_acc, col_offset * shape.M + (row_offset + COOP_TILE_SIZE), shape.M);
    MatACC acc10 = MatACC.Load<COL_MAJ>(mat_acc, (col_offset + COOP_TILE_SIZE) * shape.M + row_offset, shape.M);
    MatACC acc11 = MatACC.Load<COL_MAJ>(mat_acc, (col_offset + COOP_TILE_SIZE) * shape.M + (row_offset + COOP_TILE_SIZE), shape.M);

    for (uint t = 0; t < shape.K; t += COOP_TILE_SIZE) {
        // A: fix x=row_offset, slide y=t
        // B: fix y=col_offset, slide x=t
        MatA a0 = MatA.Load<COL_MAJ>(mat_a, t * shape.M + row_offset, shape.M);
        MatA a1 = MatA.Load<COL_MAJ>(mat_a, t * shape.M + (row_offset + COOP_TILE_SIZE), shape.M);
        MatB b0 = MatB.Load<COL_MAJ>(mat_b, col_offset * shape.K + t, shape.K);
        MatB b1 = MatB.Load<COL_MAJ>(mat_b, (col_offset + COOP_TILE_SIZE) * shape.K + t, shape.K);
        acc00 = linalg::coopMatMulAdd<float, false>(a0, b0, acc00);
        acc01 = linalg::coopMatMulAdd<float, false>(a1, b0, acc01);
        acc10 = linalg::coopMatMulAdd<float, false>(a0, b1, acc10);
        acc11 = linalg::coopMatMulAdd<float, false>(a1, b1, acc11);
    }

    acc00.Store<COL_MAJ>(mat_acc, col_offset * shape.M + row_offset, shape.M);
    acc01.Store<COL_MAJ>(mat_acc, col_offset * shape.M + (row_offset + COOP_TILE_SIZE), shape.M);
    acc10.Store<COL_MAJ>(mat_acc, (col_offset + COOP_TILE_SIZE) * shape.M + row_offset, shape.M);
    acc11.Store<COL_MAJ>(mat_acc, (col_offset + COOP_TILE_SIZE) * shape.M + (row_offset + COOP_TILE_SIZE), shape.M);
}


