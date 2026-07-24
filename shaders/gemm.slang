#define TILE_SIZE 32

[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<half> mat_A;

[[vk_binding(1, 0)]]
StructuredBuffer<half> mat_B;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float> mat_C;

/**
 * Matrix A:   M × K
 * Matrix B:   K × N
 * Matrix out: M × N
 */
struct Shape {
    uint M;
    uint N;
    uint K;
};

[[vk_push_constant]]
ConstantBuffer<Shape> shape;

// col major
[[ForceInline]]
inline half get_A_or_0(uint x, uint y) {
    return x < shape.M && y < shape.K ? mat_A[y * shape.M + x] : 0.hf;
}
[[ForceInline]]
inline half get_B_or_0(uint x, uint y) {
    return x < shape.K && y < shape.N ? mat_B[y * shape.K + x] : 0.hf;
}

[[ForceInline]]
inline void set_C_safe(uint x, uint y, float value) {
    if (x < shape.M && y < shape.N)
        mat_C[y * shape.M + x] = value;
}

groupshared half As[TILE_SIZE][TILE_SIZE + 1];
groupshared half Bs[TILE_SIZE][TILE_SIZE + 1];

[[shader("compute")]]
[[numthreads(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z)]]
void main(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID) {
    uint offset_x = gid.x * TILE_SIZE;
    uint offset_y = gid.y * TILE_SIZE;
    float acc = 0.f;

    for (uint t = 0; t < shape.K; t += TILE_SIZE) {
        uint ax = offset_x + tid.x, ay = t + tid.y;
        uint bx = t + tid.x, by = offset_y + tid.y;

        // copy to group shared memory
        As[tid.x][tid.y] = get_A_or_0(ax, ay);
        Bs[tid.x][tid.y] = get_B_or_0(bx, by);
        GroupMemoryBarrierWithGroupSync();

        // calc dot product
        [[ForceUnroll]]
        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += As[tid.x][i] * Bs[i][tid.y];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint cx = offset_x + tid.x, cy = offset_y + tid.y;
    set_C_safe(cx, cy, acc);
}

