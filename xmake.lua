add_rules("mode.debug", "mode.release")
set_policy("build.warning", true)
set_warnings("all", "extra")

if (is_plat("linux")) then
    add_requires("libsdl3", {configs = {wayland = true}})
else
    add_requires("libsdl3", {system = false})
end

add_requires("glm", "vulkan-hpp", "vulkan-memory-allocator", "eigen", "openmp")

includes("third_party/")

target("learn_vulkan", function()
    set_kind("binary")
    set_languages("c17", "c++23")
    add_files("src/**.cpp")
    add_packages("libsdl3", "glm", "vulkan-hpp", "vulkan-memory-allocator", "eigen", "openmp")
    add_deps("imgui_vulkan_sdl3", "stb")

    add_defines("VK_NO_PROTOTYPES")
    add_defines("VULKAN_HPP_NO_CONSTRUCTORS")
    add_defines("VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1")
    add_defines("VMA_STATIC_VULKAN_FUNCTIONS=0",
                "VMA_DYNAMIC_VULKAN_FUNCTIONS=1")
    add_defines("GLM_FORCE_DEPTH_ZERO_TO_ONE", "GLM_ENABLE_EXPERIMENTAL")
    on_load(function(target)
        if (is_plat("mingw")) then target:add("links", "stdc++exp") end
    end)
end)
