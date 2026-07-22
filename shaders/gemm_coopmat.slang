#define TILE_SIZE 16
#define SUBGROUP_SIZE 32

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
[[numthreads(SUBGROUP_SIZE, 1, 1)]]
void main(uint3 gid: SV_GroupID) {
    uint offset_x = gid.x * TILE_SIZE;
    uint offset_y = gid.y * TILE_SIZE;

    MatACC acc = MatACC.Load<COL_MAJ>(
        mat_acc, offset_y * shape.M + offset_x, shape.M
    );

    for (uint t = 0; t < shape.K; t += TILE_SIZE) {
        // A: fix x=offset_x, slide y=t
        // B: fix y=offset_y, slide x=t
        MatA a = MatA.Load<COL_MAJ>(mat_a, t * shape.M + offset_x, shape.M);
        MatB b = MatB.Load<COL_MAJ>(mat_b, offset_y * shape.K + t, shape.K);
        acc = linalg::coopMatMulAdd<float, false>(a, b, acc);
    }

    acc.Store<COL_MAJ>(mat_acc, offset_y * shape.M + offset_x, shape.M);
}


