add_rules("mode.debug", "mode.release")
set_policy("build.warning", true)
set_warnings("all", "extra")

-- if (is_plat("linux")) then
--     add_requires("libsdl3", {configs = {wayland = true}})
-- else
--     add_requires("libsdl3", {system = false})
-- end

option("cuBlas", function () 
    set_description("Enable cuBLAS for reference")
    set_default(true)
    add_defines("CUBLAS") 
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cublas", "cudart")
end)


add_requires("vulkan-hpp", "vulkan-memory-allocator", "eigen", "openmp")

target("learn_vulkan", function()
    set_kind("binary")
    set_languages("c17", "c++23")
    add_files("src/*.cpp")
    add_files("src/core/**.cpp")
    add_files("src/cuda_ref/*.cpp")
    add_packages("vulkan-hpp", "vulkan-memory-allocator", "eigen", "openmp")
    add_options("cuBlas")

    add_defines("VK_NO_PROTOTYPES")
    add_defines("VULKAN_HPP_NO_CONSTRUCTORS")
    add_defines("VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1")
    add_defines("VMA_STATIC_VULKAN_FUNCTIONS=0",
                "VMA_DYNAMIC_VULKAN_FUNCTIONS=1")
end)
