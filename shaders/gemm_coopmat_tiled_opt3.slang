/**
 * This file contains AI-generated code.
 */

/**
 * Fast path for column-major GEMM:
 *
 *   A:   M x K, FP16
 *   B:   K x N, FP16
 *   ACC: M x N, FP32
 *
 * Requirements:
 *   - M, N and K are multiples of 64.
 *   - The subgroup size is exactly 32.
 *   - The workgroup contains exactly 8 subgroups (256 threads).
 *
 * One workgroup computes a 64x64 output tile. Each subgroup owns a 32x16
 * region represented by two vertically adjacent 16x16 accumulators.
 */

#define COOP_TILE_SIZE      16
#define WORKGROUP_TILE_SIZE 64
#define TILE_ELEMENT_COUNT  (WORKGROUP_TILE_SIZE * WORKGROUP_TILE_SIZE)

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

// The matrices are guaranteed to be aligned and evenly divisible, so only the
// packed 32-bit input views are needed.
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

groupshared half s_a[TILE_ELEMENT_COUNT];
groupshared half s_b[TILE_ELEMENT_COUNT];

/**
 * Load one 64x64 A panel and one 64x64 B panel.
 *
 * Every iteration performs one 32-bit load from A and B. The linear shared
 * index is always even, so each half2 maps to two adjacent rows in one column.
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

        half2 a_values = mat_a[a_index >> 1];
        half2 b_values = mat_b[b_index >> 1];

        // CoopMat.Load expects scalar half arrays.
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
 * Initialize an accumulator fragment from global C so the kernel preserves
 * the existing C += A*B behavior without a separate scalar output pass.
 */
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
    // Fixed 32-lane subgroups make the subgroup mapping simple integer shifts.
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

        // Prevent the next panel load from overwriting data still being read.
        GroupMemoryBarrierWithGroupSync();
    }

    store_accumulator(acc_top, global_row, global_col);
    store_accumulator(
        acc_bottom,
        global_row + COOP_TILE_SIZE,
        global_col
    );
}
