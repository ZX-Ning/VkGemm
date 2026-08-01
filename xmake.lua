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
    "slang-static v2026.14"
)

target("core", function()
    set_kind("object")
    set_languages("c17", "c++23")
    add_files("src/**.cpp|test/**")
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

target("cublas_ref", function()
    set_kind("object")
    set_languages("c17", "c++23")
    add_packages("eigen", "fmt")
    add_files("src/cuda_ref/*.cpp")
    add_includedirs(path.join(get_config("cuda"), "include"))
    add_linkdirs(path.join(get_config("cuda"), "lib64"))
    add_defines("CUBLAS")
    add_links("cublas", "cudart")
end)

option("cuBLAS", function()
    set_description("Enable cuBLAS for reference")
    set_default(true)
    add_defines("CUBLAS")
end)

target("test_gemm", function()
    add_options("cuBLAS")
    set_kind("binary")
    set_languages("c17", "c++23")
    add_deps("core")
    add_files("src/test/gemm.cpp")
    on_config(function(target)
        if is_config("cuBLAS", true) then
            target:add("deps", "cublas_ref")
        end
    end)
end)

target("test_gemv", function()
    set_kind("binary")
    set_languages("c17", "c++23")
    add_deps("core")
    add_files("src/test/gemv.cpp")
end)
