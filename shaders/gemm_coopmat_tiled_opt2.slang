/**
 * This file contains AI-generated code.
 */

#define COOP_TILE_SIZE      16
#define WORKGROUP_TILE_SIZE 64
#define TILE_ELEMENT_COUNT  (WORKGROUP_TILE_SIZE * WORKGROUP_TILE_SIZE)

// Each storage buffer has a packed view for full tiles and a scalar view for
// boundary handling. Both views intentionally refer to the same Vulkan binding.
#pragma warning(disable: 39001)

constexpr static int SUBGROUP_TILE_ROWS = COOP_TILE_SIZE * 2;
constexpr static int SUBGROUP_TILE_COLS = COOP_TILE_SIZE;
constexpr static int SUBGROUP_GRID_ROWS =
    WORKGROUP_TILE_SIZE / SUBGROUP_TILE_ROWS;
constexpr static int PACKED_HALF_COUNT = 2;
constexpr static int PACKED_FLOAT_COUNT = 4;

[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half2> mat_a_packed;
[[vk_binding(0, 0)]]
StructuredBuffer<half> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<half2> mat_b_packed;
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

groupshared half s_a[TILE_ELEMENT_COUNT];
groupshared half s_b[TILE_ELEMENT_COUNT];
groupshared float s_c[TILE_ELEMENT_COUNT];

/**
 * Cooperatively load one 64x64 A panel and one 64x64 B panel.
 *
 * A/B are column-major, so each half2 contains two adjacent rows. Full,
 * aligned panels use one 32-bit load; boundary panels fall back to two
 * independently checked scalar loads.
 */
[[ForceInline]]
inline void load_input_panels_to_shared(
    uint row_offset,
    uint col_offset,
    uint k_offset,
    uint local_index,
    uint thread_count
) {
    bool packed_a_panel =
        (shape.M % PACKED_HALF_COUNT == 0) &&
        (row_offset + WORKGROUP_TILE_SIZE <= shape.M) &&
        (k_offset + WORKGROUP_TILE_SIZE <= shape.K);
    bool packed_b_panel =
        (shape.K % PACKED_HALF_COUNT == 0) &&
        (k_offset + WORKGROUP_TILE_SIZE <= shape.K) &&
        (col_offset + WORKGROUP_TILE_SIZE <= shape.N);

    for (uint move = local_index * PACKED_HALF_COUNT;
         move < TILE_ELEMENT_COUNT;
         move += thread_count * PACKED_HALF_COUNT) {
        uint move_row = move % WORKGROUP_TILE_SIZE;
        uint move_col = move / WORKGROUP_TILE_SIZE;

        uint a_row = row_offset + move_row;
        uint a_col = k_offset + move_col;
        uint a_index = a_col * shape.M + a_row;
        half2 a_values = packed_a_panel
            ? mat_a_packed[a_index / PACKED_HALF_COUNT]
            : half2(
                get_a_or_0(a_row, a_col),
                get_a_or_0(a_row + 1, a_col)
            );

        uint b_row = k_offset + move_row;
        uint b_col = col_offset + move_col;
        uint b_index = b_col * shape.K + b_row;
        half2 b_values = packed_b_panel
            ? mat_b_packed[b_index / PACKED_HALF_COUNT]
            : half2(
                get_b_or_0(b_row, b_col),
                get_b_or_0(b_row + 1, b_col)
            );

        // CoopMat.Load consumes scalar half arrays, so unpack the two values
        // when writing the packed global loads into shared memory.
        s_a[move] = a_values.x;
        s_a[move + 1] = a_values.y;
        s_b[move] = b_values.x;
        s_b[move + 1] = b_values.y;
    }
}

[[ForceInline]]
inline MatA load_a_from_shared(uint shared_row, uint shared_k) {
    return MatA.Load<COL_MAJ>(
        s_a,
        shared_row + shared_k * WORKGROUP_TILE_SIZE,
        WORKGROUP_TILE_SIZE
    );
}

[[ForceInline]]
inline MatB load_b_from_shared(uint shared_col, uint shared_k) {
    return MatB.Load<COL_MAJ>(
        s_b,
        shared_k + shared_col * WORKGROUP_TILE_SIZE,
        WORKGROUP_TILE_SIZE
    );
}

/**
 * Store the two vertically adjacent accumulator fragments into the 64x64
 * column-major shared output tile.
 */
[[ForceInline]]
inline void store_accumulators_to_shared(
    MatACC acc_top,
    MatACC acc_bottom,
    uint shared_row,
    uint shared_col
) {
    uint base = shared_col * WORKGROUP_TILE_SIZE + shared_row;
    acc_top.Store<COL_MAJ>(s_c, base, WORKGROUP_TILE_SIZE);
    acc_bottom.Store<COL_MAJ>(
        s_c,
        base + COOP_TILE_SIZE,
        WORKGROUP_TILE_SIZE
    );
}

/**
 * Add the completed shared output tile to the global accumulator.
 *
 * Full aligned tiles use float4 vector updates. Edge tiles use scalar guarded
 * updates so the CoopMat store itself never needs per-element bounds checks.
 */
[[ForceInline]]
inline void add_shared_output_to_global(
    uint row_offset,
    uint col_offset,
    uint local_index,
    uint thread_count
) {
    bool packed_acc_tile =
        (shape.M % PACKED_FLOAT_COUNT == 0) &&
        (row_offset + WORKGROUP_TILE_SIZE <= shape.M) &&
        (col_offset + WORKGROUP_TILE_SIZE <= shape.N);

    for (uint move = local_index * PACKED_FLOAT_COUNT;
         move < TILE_ELEMENT_COUNT;
         move += thread_count * PACKED_FLOAT_COUNT) {
        uint move_row = move % WORKGROUP_TILE_SIZE;
        uint move_col = move / WORKGROUP_TILE_SIZE;
        uint acc_row = row_offset + move_row;
        uint acc_col = col_offset + move_col;

        if (packed_acc_tile) {
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
            add_acc_safe(acc_row, acc_col, s_c[move]);
            add_acc_safe(acc_row + 1, acc_col, s_c[move + 1]);
            add_acc_safe(acc_row + 2, acc_col, s_c[move + 2]);
            add_acc_safe(acc_row + 3, acc_col, s_c[move + 3]);
        }
    }
}

/**
 * Eight subgroups cover one 64x64 output tile:
 *   subgroup grid: 2 rows x 4 columns
 *   subgroup tile: 32 rows x 16 columns
 *
 * The specialized local size must therefore contain eight subgroups
 * (for example, 32 x 4 x 2 threads on NVIDIA).
 */
[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID, uint tidx: SV_GroupIndex) {
    uint thread_count = NUM_THREADS_X * NUM_THREADS_Y * NUM_THREADS_Z;

    uint subgroup_index = WaveGetWaveIndex();
    uint subgroup_row = subgroup_index % SUBGROUP_GRID_ROWS;
    uint subgroup_col = subgroup_index / SUBGROUP_GRID_ROWS;

    uint row_offset = gid.x * WORKGROUP_TILE_SIZE;
    uint col_offset = gid.y * WORKGROUP_TILE_SIZE;
    uint shared_row = subgroup_row * SUBGROUP_TILE_ROWS;
    uint shared_col = subgroup_col * SUBGROUP_TILE_COLS;

    // Each subgroup computes two vertically adjacent 16x16 fragments.
    var acc_top = MatACC(0.f);
    var acc_bottom = MatACC(0.f);

    for (uint k_offset = 0;
         k_offset < shape.K;
         k_offset += WORKGROUP_TILE_SIZE) {
        load_input_panels_to_shared(
            row_offset,
            col_offset,
            k_offset,
            tidx,
            thread_count
        );
        GroupMemoryBarrierWithGroupSync();

        [[ForceUnroll]]
        for (uint shared_k = 0;
             shared_k < WORKGROUP_TILE_SIZE;
             shared_k += COOP_TILE_SIZE) {
            MatA a_top = load_a_from_shared(shared_row, shared_k);
            MatA a_bottom =
                load_a_from_shared(shared_row + COOP_TILE_SIZE, shared_k);
            MatB b = load_b_from_shared(shared_col, shared_k);

            acc_top =
                linalg::coopMatMulAdd<float, false>(a_top, b, acc_top);
            acc_bottom =
                linalg::coopMatMulAdd<float, false>(a_bottom, b, acc_bottom);
        }

        // All subgroups must finish reading the panels before the next
        // iteration overwrites s_a and s_b.
        GroupMemoryBarrierWithGroupSync();
    }

    store_accumulators_to_shared(
        acc_top,
        acc_bottom,
        shared_row,
        shared_col
    );
    GroupMemoryBarrierWithGroupSync();

    add_shared_output_to_global(
        row_offset,
        col_offset,
        tidx,
        thread_count
    );
}
