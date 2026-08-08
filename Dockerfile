FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build

RUN apt-get update \
    && apt-get install -y \
    libxcb1-dev vulkan-tools libglvnd0 libgl1 libglx0 build-essential cmake ninja-build curl wget nvtop

RUN curl -fsSL https://xmake.io/shget.text | bash

RUN apt-get clean -y

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV VULKAN_SDK_VERSION=1.4.357.0

RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/latest/linux/vulkan_sdk.tar.gz \
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \ 
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan

WORKDIR /root/workspace/vkgemm

COPY . .

ENV XMAKE_ROOT=y

RUN apt-get install -y clang unzip

RUN ~/.local/bin/xmake f --cuda=/usr/local/cuda --toolchain=clang --cu=clang++ --culd=clang++ -m release -y

RUN ~/.local/bin/xmake -v

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y \
    libgomp1 libegl1 libxcb1-dev vulkan-tools openssh-server vim

RUN apt-get clean -y

RUN touch /root/.ssh/known_hosts && touch /root/.ssh/authorized_keys

RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && chmod 644 /root/.ssh/known_hosts

WORKDIR /root/workspace

COPY --from=build /root/workspace/vkgemm/build/linux/x86_64/release/* ./
COPY ./shaders ./shaders

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["bash"]