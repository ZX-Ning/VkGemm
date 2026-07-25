/**
 * This file contains AI-generated code.
 */
/**
 * opt3 plus a skewed shared-memory layout.
 *
 * Requirements:
 *   - M, N and K are multiples of 64.
 *   - The subgroup size is exactly 32.
 *   - The workgroup contains exactly 8 subgroups (256 threads).
 *
 * One workgroup computes a 64x64 output tile. Each subgroup owns a 32x16
 * region represented by two vertically adjacent 16x16 accumulators.
 */

#define COOP_TILE_SIZE       16
#define WORKGROUP_TILE_SIZE  64
#define TILE_ELEMENT_COUNT   (WORKGROUP_TILE_SIZE * WORKGROUP_TILE_SIZE)
#define SHARED_SKEW          8
#define SHARED_STRIDE        (WORKGROUP_TILE_SIZE + SHARED_SKEW)
#define SHARED_ELEMENT_COUNT (SHARED_STRIDE * WORKGROUP_TILE_SIZE)

constexpr static uint SUBGROUP_SIZE = 32;
constexpr static uint SUBGROUP_COUNT = 8;
constexpr static uint GROUP_THREAD_COUNT = SUBGROUP_SIZE * SUBGROUP_COUNT;
constexpr static uint SUBGROUP_TILE_ROWS = COOP_TILE_SIZE * 2;
constexpr static uint SUBGROUP_TILE_COLS = COOP_TILE_SIZE;
constexpr static uint PACKED_HALF_COUNT = 2;

[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half2> mat_a;

[[vk_binding(1, 0)]]
StructuredBuffer<half2> mat_b;

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

const static let COL_MAJ = linalg::CoopMatMatrixLayout::ColumnMajor;

// A stride of 72 half values is 144 bytes. Adjacent columns therefore move by
// four 4-byte banks instead of wrapping to the same bank as stride 64 does.
// The 144-byte stride also preserves the 16-byte alignment required by the
// cooperative-matrix shared loads used below.
groupshared half s_a[SHARED_ELEMENT_COUNT];
groupshared half s_b[SHARED_ELEMENT_COUNT];

/**
 * Load logical 64x64 panels into the physical 72x64 shared arrays.
 */
[[ForceInline]]
inline void load_input_panels_to_shared(
    uint row_offset,
    uint col_offset,
    uint k_offset,
    uint local_index
) {
    for (uint move = local_index * PACKED_HALF_COUNT;
         move < TILE_ELEMENT_COUNT;
         move += GROUP_THREAD_COUNT * PACKED_HALF_COUNT) {
        uint move_row = move & (WORKGROUP_TILE_SIZE - 1);
        uint move_col = move >> 6;

        uint a_index =
            (k_offset + move_col) * shape.M + row_offset + move_row;
        uint b_index =
            (col_offset + move_col) * shape.K + k_offset + move_row;
        // `move` already contains move_col * 64; add eight padding elements
        // for every preceding column to obtain the physical shared index.
        uint shared_index = move + (move_col << 3);

        half2 a_values = mat_a[a_index >> 1];
        half2 b_values = mat_b[b_index >> 1];

        s_a[shared_index] = a_values.x;
        s_a[shared_index + 1] = a_values.y;
        s_b[shared_index] = b_values.x;
        s_b[shared_index + 1] = b_values.y;
    }
}

[[ForceInline]]
inline MatA load_a_from_shared(uint shared_row, uint shared_k) {
    return MatA.Load<COL_MAJ>(
        s_a,
        shared_row + shared_k * SHARED_STRIDE,
        SHARED_STRIDE
    );
}

[[ForceInline]]
inline MatB load_b_from_shared(uint shared_col, uint shared_k) {
    return MatB.Load<COL_MAJ>(
        s_b,
        shared_k + shared_col * SHARED_STRIDE,
        SHARED_STRIDE
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

    uint row_offset = group_id.x * WORKGROUP_TILE_SIZE;
    uint col_offset = group_id.y * WORKGROUP_TILE_SIZE;
    uint shared_row = subgroup_row * SUBGROUP_TILE_ROWS;
    uint shared_col = subgroup_col * SUBGROUP_TILE_COLS;
    uint global_row = row_offset + shared_row;
    uint global_col = col_offset + shared_col;

    var acc_top = load_accumulator(global_row, global_col);
    var acc_bottom =
        load_accumulator(global_row + COOP_TILE_SIZE, global_col);

    for (uint k_offset = 0;
         k_offset < shape.K;
         k_offset += WORKGROUP_TILE_SIZE) {
        load_input_panels_to_shared(
            row_offset,
            col_offset,
            k_offset,
            local_index
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

        GroupMemoryBarrierWithGroupSync();
    }

    store_accumulator(acc_top, global_row, global_col);
    store_accumulator(
        acc_bottom,
        global_row + COOP_TILE_SIZE,
        global_col
    );
}
