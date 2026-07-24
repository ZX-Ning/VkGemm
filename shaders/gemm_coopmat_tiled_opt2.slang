/**
 * This file contains AI-generated code.
 */

#define COOP_TILE_SIZE     16
#define COOP_TILES         4
#define LARGE_TILE_SIZE    64
#define TILE_ELEMENT_COUNT (LARGE_TILE_SIZE * LARGE_TILE_SIZE)

// Each storage buffer has a packed view for full tiles and a scalar view for
// boundary handling. Both views intentionally refer to the same Vulkan binding.
#pragma warning(disable: 39001)

constexpr static int SUB_TILES = (COOP_TILES / 2);
constexpr static int SUB_TILE_SIZE = COOP_TILE_SIZE * 2;
constexpr static int PACKED_HALF_COUNT = 4;
constexpr static int PACKED_FLOAT_COUNT = 4;

[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half4> mat_a_packed;
[[vk_binding(0, 0)]]
StructuredBuffer<half> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<half4> mat_b_packed;
[[vk_binding(1, 0)]]
StructuredBuffer<half> mat_b;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float4> mat_acc_packed;
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
groupshared half s_a[TILE_ELEMENT_COUNT];
groupshared half s_b[TILE_ELEMENT_COUNT];
groupshared float s_c[TILE_ELEMENT_COUNT];

/**
 * Assume numthreads = (COOP_TILES / 2)^2 * SUBGROUPSIZE
 * 4 * 32 on nvidia
 */
[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID, uint tidx: SV_GroupIndex) {
    uint thread_count = NUM_THREADS_X * NUM_THREADS_Y * NUM_THREADS_Z;

    uint sg_index = WaveGetWaveIndex();
    uint2 sgid = uint2(sg_index % SUB_TILES, sg_index / SUB_TILES);

    uint row_offset = gid.x * LARGE_TILE_SIZE;
    uint col_offset = gid.y * LARGE_TILE_SIZE;

    uint shared_row_offset = sgid.x * SUB_TILE_SIZE;
    uint shared_col_offset = sgid.y * SUB_TILE_SIZE;

    // A subgroup work on SUB_TILE_SIZE x SUB_TILE_SIZE matrix
    // which is 2x2 coopmat
    var acc00 = MatACC(0.f);
    var acc01 = MatACC(0.f);
    var acc10 = MatACC(0.f);
    var acc11 = MatACC(0.f);

    for (uint t = 0; t < shape.K; t += LARGE_TILE_SIZE) {
        // Column-major elements are contiguous along the row dimension.
        // The packed view is valid only when:
        //   1. four rows fit inside the current panel; and
        //   2. the matrix leading dimension keeps every chunk 8-byte aligned.
        //
        // A uses M as its leading dimension, while B uses K.
        bool packed_a_panel =
            (shape.M % PACKED_HALF_COUNT == 0) &&
            (row_offset + LARGE_TILE_SIZE <= shape.M) &&
            (t + LARGE_TILE_SIZE <= shape.K);
        bool packed_b_panel =
            (shape.K % PACKED_HALF_COUNT == 0) &&
            (t + LARGE_TILE_SIZE <= shape.K) &&
            (col_offset + LARGE_TILE_SIZE <= shape.N);

        // Cooperative global -> shared copy.
        //
        // `move` is always a multiple of four. Since LARGE_TILE_SIZE is also a
        // multiple of four, one iteration always covers four adjacent rows in
        // one column and never crosses a shared-memory column boundary.
        for (uint move = tidx * PACKED_HALF_COUNT;
             move < TILE_ELEMENT_COUNT;
             move += thread_count * PACKED_HALF_COUNT) {
            uint move_row = move % LARGE_TILE_SIZE;
            uint move_col = move / LARGE_TILE_SIZE;

            uint a_row = row_offset + move_row;
            uint a_col = t + move_col;
            uint a_index = a_col * shape.M + a_row;
            half4 a_values;
            if (packed_a_panel) {
                // One 8-byte global load.
                a_values = mat_a_packed[a_index / PACKED_HALF_COUNT];
            }
            else {
                // Edge panels use scalar reads so each element can be checked
                // independently and out-of-range values can be zero-filled.
                a_values = half4(
                    get_a_or_0(a_row, a_col),
                    get_a_or_0(a_row + 1, a_col),
                    get_a_or_0(a_row + 2, a_col),
                    get_a_or_0(a_row + 3, a_col)
                );
            }

            uint b_row = t + move_row;
            uint b_col = col_offset + move_col;
            uint b_index = b_col * shape.K + b_row;
            half4 b_values;
            if (packed_b_panel) {
                // One 8-byte global load.
                b_values = mat_b_packed[b_index / PACKED_HALF_COUNT];
            }
            else {
                // Same boundary-safe path as A.
                b_values = half4(
                    get_b_or_0(b_row, b_col),
                    get_b_or_0(b_row + 1, b_col),
                    get_b_or_0(b_row + 2, b_col),
                    get_b_or_0(b_row + 3, b_col)
                );
            }

            // CoopMat.Load consumes a scalar half array, so unpack the loaded
            // vectors while writing them into shared memory.
            s_a[move] = a_values.x;
            s_a[move + 1] = a_values.y;
            s_a[move + 2] = a_values.z;
            s_a[move + 3] = a_values.w;
            s_b[move] = b_values.x;
            s_b[move + 1] = b_values.y;
            s_b[move + 2] = b_values.z;
            s_b[move + 3] = b_values.w;
        }

        GroupMemoryBarrierWithGroupSync();

        [[ForceUnroll]]  // loop 4 times
        for (uint sub_t = 0; sub_t < LARGE_TILE_SIZE; sub_t += COOP_TILE_SIZE) {
            var a0 = MatA.Load<COL_MAJ>(
                s_a,
                shared_row_offset + sub_t * LARGE_TILE_SIZE,
                LARGE_TILE_SIZE
            );
            var a1 = MatA.Load<COL_MAJ>(
                s_a,
                (shared_row_offset + COOP_TILE_SIZE) + sub_t * LARGE_TILE_SIZE,
                LARGE_TILE_SIZE
            );
            var b0 = MatB.Load<COL_MAJ>(
                s_b,
                sub_t + shared_col_offset * LARGE_TILE_SIZE,
                LARGE_TILE_SIZE
            );
            var b1 = MatB.Load<COL_MAJ>(
                s_b,
                sub_t + (shared_col_offset + COOP_TILE_SIZE) * LARGE_TILE_SIZE,
                LARGE_TILE_SIZE
            );
            acc00 = linalg::coopMatMulAdd<float, false>(a0, b0, acc00);
            acc01 = linalg::coopMatMulAdd<float, false>(a1, b0, acc01);
            acc10 = linalg::coopMatMulAdd<float, false>(a0, b1, acc10);
            acc11 = linalg::coopMatMulAdd<float, false>(a1, b1, acc11);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Materialize the four accumulator fragments as one column-major 64x64
    // shared tile. This lets all workgroup threads perform the final global
    // write cooperatively and keeps boundary checks out of CoopMat.Store.
    acc00.Store<COL_MAJ>(
        s_c,
        shared_col_offset * LARGE_TILE_SIZE + shared_row_offset,
        LARGE_TILE_SIZE
    );
    acc01.Store<COL_MAJ>(
        s_c,
        shared_col_offset * LARGE_TILE_SIZE + (shared_row_offset + COOP_TILE_SIZE),
        LARGE_TILE_SIZE
    );
    acc10.Store<COL_MAJ>(
        s_c,
        (shared_col_offset + COOP_TILE_SIZE) * LARGE_TILE_SIZE + shared_row_offset,
        LARGE_TILE_SIZE
    );
    acc11.Store<COL_MAJ>(
        s_c,
        (shared_col_offset + COOP_TILE_SIZE) * LARGE_TILE_SIZE + (shared_row_offset + COOP_TILE_SIZE),
        LARGE_TILE_SIZE
    );
    GroupMemoryBarrierWithGroupSync();

    // A float4 maps to four adjacent rows. The packed output path therefore
    // requires a complete tile and an M stride divisible by four.
    bool packed_acc_tile =
        (shape.M % PACKED_FLOAT_COUNT == 0) &&
        (row_offset + LARGE_TILE_SIZE <= shape.M) &&
        (col_offset + LARGE_TILE_SIZE <= shape.N);

    for (uint move = tidx * PACKED_FLOAT_COUNT;
         move < TILE_ELEMENT_COUNT;
         move += thread_count * PACKED_FLOAT_COUNT) {
        uint move_row = move % LARGE_TILE_SIZE;
        uint move_col = move / LARGE_TILE_SIZE;
        uint acc_row = row_offset + move_row;
        uint acc_col = col_offset + move_col;

        if (packed_acc_tile) {
            // mat_acc uses += semantics. This typed view emits one vector load,
            // one vector add, and one 16-byte vector store.
            uint acc_index = acc_col * shape.M + acc_row;
            float4 values = float4(
                s_c[move],
                s_c[move + 1],
                s_c[move + 2],
                s_c[move + 3]
            );
            mat_acc_packed[acc_index / PACKED_FLOAT_COUNT] += values;
        }
        else {
            // Boundary tiles fall back to independently guarded scalar updates.
            add_acc_safe(acc_row, acc_col, s_c[move]);
            add_acc_safe(acc_row + 1, acc_col, s_c[move + 1]);
            add_acc_safe(acc_row + 2, acc_col, s_c[move + 2]);
            add_acc_safe(acc_row + 3, acc_col, s_c[move + 3]);
        }
    }
}

