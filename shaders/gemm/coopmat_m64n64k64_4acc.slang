/**
 * Cooperative-matrix GEMM with a 64x64x64 workgroup tile.
 *
 * Requirements:
 *   - M, N and K are multiples of 64.
 *   - subgroup size is 32.
 *   - numthreads is (32, 2, 2): 4 subgroups, 128 threads.
 *
 * A workgroup produces 64x64. Each subgroup produces 32x32 with four
 * 16x16 accumulators. K is staged in chunks of 64.
 */

#define COOP_TILE_SIZE   16
#define OUTPUT_TILE_SIZE 64
#define K_TILE_SIZE      64
#define SHARED_SKEW      8
#define SHARED_K_STRIDE  (K_TILE_SIZE + SHARED_SKEW)
#define LOAD_HALF_COUNT  8
#define LOAD_WORD_COUNT  (LOAD_HALF_COUNT / 2)
#define PANEL_VECTOR_COUNT                             \
    (OUTPUT_TILE_SIZE * K_TILE_SIZE / LOAD_HALF_COUNT)

static const uint SUBGROUP_SIZE = 32;
static const uint SUBGROUP_COUNT = 4;
static const uint GROUP_THREAD_COUNT =
    SUBGROUP_SIZE * SUBGROUP_COUNT;
static const uint SUBGROUP_TILE_SIZE = COOP_TILE_SIZE * 2;

[[vk::constant_id(0)]]
const uint NUM_THREADS_X;

[[vk::constant_id(1)]]
const uint NUM_THREADS_Y;

[[vk::constant_id(2)]]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<uint4> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<uint4> mat_b;

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

const static let ROW_MAJ =
    linalg::CoopMatMatrixLayout::RowMajor;
const static let COL_MAJ =
    linalg::CoopMatMatrixLayout::ColumnMajor;

// A is transposed while staging so both cooperative loads traverse K
// contiguously. Eight padding halfs reduce shared-memory bank conflicts.
groupshared half s_a[OUTPUT_TILE_SIZE * SHARED_K_STRIDE];
groupshared half s_b[OUTPUT_TILE_SIZE * SHARED_K_STRIDE];

[[ForceInline]]
inline void load_input_panels_to_shared(
    uint row_offset,
    uint col_offset,
    uint k_offset,
    uint local_index
) {
    // Each iteration moves eight half values (16 bytes) from both inputs.
    for (uint vector_index = local_index;
         vector_index < PANEL_VECTOR_COUNT;
         vector_index += GROUP_THREAD_COUNT) {
        // Global A is column-major. Scatter its eight adjacent rows into the
        // row-major shared panel [M][K].
        uint a_row_group =
            vector_index % (OUTPUT_TILE_SIZE / LOAD_HALF_COUNT);
        uint a_row = a_row_group * LOAD_HALF_COUNT;
        uint a_k =
            vector_index / (OUTPUT_TILE_SIZE / LOAD_HALF_COUNT);
        uint4 a_words = mat_a[
            ((k_offset + a_k) * shape.M + row_offset + a_row) /
            LOAD_HALF_COUNT
        ];

        // Global B and shared B are both column-major, with K contiguous.
        uint b_k_group =
            vector_index % (K_TILE_SIZE / LOAD_HALF_COUNT);
        uint b_k = b_k_group * LOAD_HALF_COUNT;
        uint b_col =
            vector_index / (K_TILE_SIZE / LOAD_HALF_COUNT);
        uint4 b_words = mat_b[
            ((col_offset + b_col) * shape.K + k_offset + b_k) /
            LOAD_HALF_COUNT
        ];

        [[ForceUnroll]]
        for (uint word = 0; word < LOAD_WORD_COUNT; ++word) {
            half2 a_values = unpackHalf2x16ToHalf(a_words[word]);
            uint a_shared_row = a_row + word * 2;
            s_a[a_shared_row * SHARED_K_STRIDE + a_k] =
                a_values.x;
            s_a[(a_shared_row + 1) * SHARED_K_STRIDE + a_k] =
                a_values.y;

            half2 b_values = unpackHalf2x16ToHalf(b_words[word]);
            uint b_shared_k = b_k + word * 2;
            s_b[b_col * SHARED_K_STRIDE + b_shared_k] =
                b_values.x;
            s_b[b_col * SHARED_K_STRIDE + b_shared_k + 1] =
                b_values.y;
        }
    }
}

[[ForceInline]]
inline MatA load_a_from_shared(uint shared_row, uint shared_k) {
    return MatA.Load<ROW_MAJ>(
        s_a,
        shared_row * SHARED_K_STRIDE + shared_k,
        SHARED_K_STRIDE
    );
}

[[ForceInline]]
inline MatB load_b_from_shared(uint shared_col, uint shared_k) {
    return MatB.Load<COL_MAJ>(
        s_b,
        shared_col * SHARED_K_STRIDE + shared_k,
        SHARED_K_STRIDE
    );
}

[[ForceInline]]
inline MatACC load_accumulator(uint global_row, uint global_col) {
    return MatACC.Load<COL_MAJ>(
        mat_acc,
        global_col * shape.M + global_row,
        shape.M
    );
}

[[ForceInline]]
inline void store_accumulator(
    MatACC acc,
    uint global_row,
    uint global_col
) {
    acc.Store<COL_MAJ>(
        mat_acc,
        global_col * shape.M + global_row,
        shape.M
    );
}

[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(
    uint3 group_id: SV_GroupID,
    uint local_index: SV_GroupIndex
) {
    uint subgroup_index = local_index >> 5;
    uint subgroup_row = subgroup_index & 1;
    uint subgroup_col = subgroup_index >> 1;

    uint row_offset = group_id.x * OUTPUT_TILE_SIZE;
    uint col_offset = group_id.y * OUTPUT_TILE_SIZE;
    uint shared_row = subgroup_row * SUBGROUP_TILE_SIZE;
    uint shared_col = subgroup_col * SUBGROUP_TILE_SIZE;
    uint global_row = row_offset + shared_row;
    uint global_col = col_offset + shared_col;

    var acc00 = load_accumulator(global_row, global_col);
    var acc10 =
        load_accumulator(global_row + COOP_TILE_SIZE, global_col);
    var acc01 =
        load_accumulator(global_row, global_col + COOP_TILE_SIZE);
    var acc11 = load_accumulator(
        global_row + COOP_TILE_SIZE,
        global_col + COOP_TILE_SIZE
    );

    for (uint k_offset = 0;
         k_offset < shape.K;
         k_offset += K_TILE_SIZE) {
        load_input_panels_to_shared(
            row_offset,
            col_offset,
            k_offset,
            local_index
        );
        GroupMemoryBarrierWithGroupSync();

        [[ForceUnroll]]
        for (uint shared_k = 0;
             shared_k < K_TILE_SIZE;
             shared_k += COOP_TILE_SIZE) {
            MatA a0 = load_a_from_shared(shared_row, shared_k);
            MatA a1 = load_a_from_shared(
                shared_row + COOP_TILE_SIZE,
                shared_k
            );
            MatB b0 = load_b_from_shared(shared_col, shared_k);
            MatB b1 = load_b_from_shared(
                shared_col + COOP_TILE_SIZE,
                shared_k
            );

            acc00 =
                linalg::coopMatMulAdd<float, false>(a0, b0, acc00);
            acc10 =
                linalg::coopMatMulAdd<float, false>(a1, b0, acc10);
            acc01 =
                linalg::coopMatMulAdd<float, false>(a0, b1, acc01);
            acc11 =
                linalg::coopMatMulAdd<float, false>(a1, b1, acc11);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    store_accumulator(acc00, global_row, global_col);
    store_accumulator(
        acc10,
        global_row + COOP_TILE_SIZE,
        global_col
    );
    store_accumulator(
        acc01,
        global_row,
        global_col + COOP_TILE_SIZE
    );
    store_accumulator(
        acc11,
        global_row + COOP_TILE_SIZE,
        global_col + COOP_TILE_SIZE
    );
}
