#define SUB_TILE_SIZE   16
#define SUB_TILES       4
#define LARGE_TILE_SIZE 64
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
    SUB_TILE_SIZE,
    SUB_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixA>;
typealias MatB = linalg::CoopMat<
    half,
    MemoryScope.Subgroup,
    SUB_TILE_SIZE,
    SUB_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixB>;
typealias MatACC = linalg::CoopMat<
    float,
    MemoryScope.Subgroup,
    SUB_TILE_SIZE,
    SUB_TILE_SIZE,
    linalg::CoopMatMatrixUse.MatrixAccumulator>;

const static let COL_MAJ = linalg::CoopMatMatrixLayout::ColumnMajor;

[[ForceInline]]
inline half get_a_or_0(uint x, uint y) {
    return (x < shape.M && y < shape.K) ? mat_a[y * shape.M + x] : 0.hf;
}
[[ForceInline]]
inline half get_b_or_0(uint x, uint y) {
    return (x < shape.K && y < shape.N) ? mat_b[y * shape.K + x] : 0.hf;
}

[[ForceInline]]
inline void add_acc_safe(uint x, uint y, float value) {
    if (x < shape.M && y < shape.N)
        mat_acc[y * shape.M + x] += value;
}

// Assume 4 subgroups per workgroup
// if subgroup size=32, tile size=16, then a workgroup will work on a 64x64 tile.
groupshared half s_a[LARGE_TILE_SIZE * LARGE_TILE_SIZE];
groupshared half s_b[LARGE_TILE_SIZE * LARGE_TILE_SIZE];
groupshared float s_c[LARGE_TILE_SIZE * LARGE_TILE_SIZE];

/**
 * Assume numthreads(SUB_TILES, SUB_TILES, SUBGROUPSIZE)
 */
[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID) {
    uint sg_index = WaveGetWaveIndex();  // assume 0-15
    uint sg_size = WaveGetLaneCount();
    uint2 sgid = uint2(sg_index % SUB_TILES, sg_index / SUB_TILES);

    uint row_offset = gid.x * LARGE_TILE_SIZE;
    uint col_offset = gid.y * LARGE_TILE_SIZE;

    uint shared_row_offset = sgid.x * SUB_TILE_SIZE;
    uint shared_col_offset = sgid.y * SUB_TILE_SIZE;

    MatACC acc;
    acc.clear();
    uint move_per_thread = SUB_TILE_SIZE * SUB_TILE_SIZE / sg_size;
    uint move_start =
        sg_index * SUB_TILE_SIZE * SUB_TILE_SIZE +
        WaveGetLaneIndex() * move_per_thread;
    for (uint t = 0; t < shape.K; t += LARGE_TILE_SIZE) {
        // cooperative load a,b to shared mem
        for (uint move_i = 0; move_i < move_per_thread; move_i++) {
            uint move = move_start + move_i;
            uint move_row = move % LARGE_TILE_SIZE;
            uint move_col = move / LARGE_TILE_SIZE;
            s_a[move] = get_a_or_0(row_offset + move_row, t + move_col);
            s_b[move] = get_b_or_0(t + move_row, col_offset + move_col);
        }

        GroupMemoryBarrierWithGroupSync();

        [[ForceUnroll]]  // loop 4 times
        for (uint sub_t = 0; sub_t < LARGE_TILE_SIZE; sub_t += SUB_TILE_SIZE) {
            MatA a = MatA.Load<COL_MAJ>(s_a, shared_row_offset + sub_t * LARGE_TILE_SIZE, LARGE_TILE_SIZE);
            MatB b = MatB.Load<COL_MAJ>(s_b, sub_t + shared_col_offset * LARGE_TILE_SIZE, LARGE_TILE_SIZE);
            acc = linalg::coopMatMulAdd<float, false>(a, b, acc);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // uint acc_row = row_offset + sgid.x * SUB_TILE_SIZE;
    // uint acc_col = col_offset + sgid.y * SUB_TILE_SIZE;
    // acc.Store<COL_MAJ>(mat_acc, acc_row + acc_col * shape.M, shape.M);
    
    acc.Store<COL_MAJ>(s_c, shared_row_offset + shared_col_offset * LARGE_TILE_SIZE, LARGE_TILE_SIZE);
    GroupMemoryBarrierWithGroupSync();
    for (uint move_i = 0; move_i < move_per_thread; move_i++) {
        uint move = move_start + move_i;
        uint move_row = move % LARGE_TILE_SIZE;
        uint move_col = move / LARGE_TILE_SIZE;
        add_acc_safe(row_offset + move_row, col_offset + move_col, s_c[move]);
    }
}