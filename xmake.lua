add_rules("mode.debug", "mode.release")
set_policy("build.warning", true)
set_warnings("all", "extra")

includes("xmake/*.lua")

-- if (is_plat("linux")) then
--     add_requires("libsdl3", {configs = {wayland = true}})
-- else
--     add_requires("libsdl3", {system = false})
-- end
add_requires(
    "vulkan-hpp",
    "vulkan-memory-allocator",
    "eigen",
    "openmp",
    "fmt",
    "argparse",
    "slang-static v2026.14"
)

target("core", function()
    set_kind("object")
    set_languages("c17", "c++23")
    add_files("src/core/**.cpp", "src/shader/**.cpp")
    add_packages(
        "vulkan-hpp",
        "vulkan-memory-allocator",
        "eigen",
        "openmp",
        "fmt",
        "slang-static",
        { public = true }
    )

    add_defines("VK_NO_PROTOTYPES", { public = true })
    add_defines("VULKAN_HPP_NO_CONSTRUCTORS", { public = true })
    add_defines("VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1", { public = true })
    add_defines(
        "VMA_STATIC_VULKAN_FUNCTIONS=0",
        "VMA_DYNAMIC_VULKAN_FUNCTIONS=1",
        { public = true }
    )
end)

target("cuda_ref", function()
    set_kind("object")
    set_languages("c++23")
    add_packages("eigen", "fmt")
    add_files("src/cuda_ref/*.cpp", "src/cuda_ref/*.cu")
    add_includedirs(path.join(get_config("cuda"), "include"))
    add_linkdirs(path.join(get_config("cuda"), "lib64"))
    add_defines("CUDA_REF")
    add_links("cublas", "cudart")
    add_cuflags("-Wno-unknown-cuda-version")
    add_cugencodes("native")
end)

option("CUDA_REF", function()
    set_description("Enable cuBLAS for reference")
    set_default(true)
    add_defines("CUDA_REF")
end)

target("test_gemm", function()
    add_options("CUDA_REF")
    set_kind("binary")
    set_languages("c17", "c++23")
    add_deps("core")
    add_packages("argparse")
    add_files("src/test/gemm.cpp")
    if is_config("CUDA_REF", true) then
        add_deps("cuda_ref")
        -- cuda_ref is an object target, so its Clang CUDA offload image is
        -- passed directly to this binary. Tell Clang's final link to package
        -- and register that image instead of treating it as a plain C++
        -- object; otherwise a kernel launch fails with
        -- cudaErrorInvalidResourceHandle.
        add_ldflags(
            "--offload-link",
            "--cuda-path=" .. get_config("cuda"),
            "-fcuda-rdc",
            "-Wno-unknown-cuda-version",
            { force = true }
        )
    end
end)

target("test_gemv", function()
    set_kind("binary")
    set_languages("c17", "c++23")
    add_deps("core")
    add_files("src/test/gemv.cpp")
end)
