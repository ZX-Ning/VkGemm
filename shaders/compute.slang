[[vk_binding(0, 0)]]
StructuredBuffer<float> vec1;

[[vk_binding(1, 0)]]
StructuredBuffer<float> vec2;

[[vk_binding(2, 0)]]
RWStructuredBuffer<float> vec_out;

[[vk_push_constant]]
ConstantBuffer<uint> vec_size;

[[shader("compute")]]
[[numthreads(256, 1, 1)]]
void main(uint idx: SV_DispatchThreadID) {
    if (idx < vec_size) {
        vec_out[idx] = vec1[idx] + vec2[idx];
    }
}


