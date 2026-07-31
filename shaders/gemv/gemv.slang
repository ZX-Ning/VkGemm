[vk::constant_id(0)]
const uint NUM_THREADS_X;

[vk::constant_id(1)]
const uint NUM_THREADS_Y;

[vk::constant_id(2)]
const uint NUM_THREADS_Z;

[[vk_binding(0, 0)]]
StructuredBuffer<float> mat_A;

[[vk_binding(1, 0)]]
StructuredBuffer<float> vec_x;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float> vec_y;

/**
 * Matrix A:   M × N
 * Vector x:   N
 */
struct Shape {
    uint M;
    uint N;
};

[[vk_push_constant]]
ConstantBuffer<Shape> shape;

[[shader("compute")]]
[[numthreads(NUM_THREADS_X, 1, 1)]]
void main(uint gidx: SV_GroupID) {
    uint sg_tid = WaveGetLaneIndex();
    uint sg_size = WaveGetLaneCount();
    uint row = (NUM_THREADS_X / sg_size) * gidx + WaveGetWaveIndex();

    uint elements = (shape.N + sg_size - 1) / sg_size;
    float acc = 0.f;

    // for A: row: sg_id, col: [sg_tid * elements, sg_tid * elements + elements )
    for (uint i = 0; i < elements; i++) {
        uint col = sg_tid * elements + i;
        if (col < shape.N) {
            acc += mat_A[row + col * shape.M] * vec_x[col];
        }
    }
    float sg_acc = WaveActiveSum(acc);
    if (sg_tid == 0) {
        vec_y[row] += sg_acc;
    }
}

