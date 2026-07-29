package("slang-static", function()
    set_homepage("https://github.com/shader-slang/slang")
    set_description("Making it easier to work with shaders")
    set_license("MIT")

    add_urls("https://github.com/shader-slang/slang.git", {submodules = false})
    add_versions("v2026.14", "4265906358862245cb25f091cb47dc426c15267b")

    add_deps("cmake")
    add_deps("miniz", "lz4", "unordered_dense", "vulkan-headers")
    
    add_links("slang-compiler", "compiler-core", "core", "cmark-gfm", "dl")

    on_install(function(package)
        import("devel.git")
        import("devel.git.submodule")
        submodule.update({
            repodir = os.curdir(),
            init = true,
            paths = {
                "external/cmark", "external/fast_float", "external/lua",
                "external/spirv-headers"
            }
        })
        local configs = {
            "-DSLANG_LIB_TYPE=STATIC", "-DSLANG_ENABLE_TESTS=OFF",
            "-DSLANG_ENABLE_EXAMPLES=OFF", "-DSLANG_ENABLE_GFX=OFF",
            "-DSLANG_ENABLE_SLANG_RHI=OFF", "-DSLANG_ENABLE_SLANGD=OFF",
            "-DSLANG_ENABLE_SLANGC=OFF", "-DSLANG_ENABLE_SLANGI=OFF",
            "-DSLANG_ENABLE_SLANGRT=OFF", "-DSLANG_ENABLE_REPLAYER=OFF",
            "-DSLANG_ENABLE_SLANG_PROXY=OFF", "-DSLANG_ENABLE_DXIL=OFF",
            "-DSLANG_ENABLE_SLANG_GLSLANG=OFF",
            "-DSLANG_ENABLE_GLSL_MODULE=OFF",
            "-DSLANG_SLANG_LLVM_FLAVOR=DISABLE", "-DSLANG_ENABLE_CUDA=OFF",
            "-DSLANG_ENABLE_OPTIX=OFF", "-DSLANG_ENABLE_NVAPI=OFF",
            "-DSLANG_ENABLE_AFTERMATH=OFF", "-DSLANG_ENABLE_XLIB=OFF",
            "-DSLANG_ENABLE_DX_ON_VK=OFF", "-DSLANG_EXCLUDE_DAWN=ON",
            "-DSLANG_EXCLUDE_TINT=ON", "-DSLANG_ENABLE_MIMALLOC=OFF",
            "-DSLANG_ENABLE_SPIRV_TOOLS_MIMALLOC=OFF",
            "-DSLANG_USE_SYSTEM_MINIZ=ON", "-DSLANG_USE_SYSTEM_LZ4=ON",
            "-DSLANG_USE_SYSTEM_UNORDERED_DENSE=ON",
            "-DSLANG_USE_SYSTEM_VULKAN_HEADERS=ON",
            "-DSLANG_EMBED_CORE_MODULE=ON",
            "-DSLANG_EMBED_CORE_MODULE_SOURCE=OFF",
            "-DSLANG_ENABLE_FULL_IR_VALIDATION=OFF",
            "-DSLANG_ENABLE_VALIDATION_VM_BYTECODE=OFF",
            "-DSLANG_ENABLE_ASAN=OFF", "-DSLANG_ENABLE_COVERAGE=OFF",
            "-DSLANG_ENABLE_TIME_TRACE=OFF",
            "-DSLANG_ENABLE_RELEASE_DEBUG_INFO=OFF",
            "-DSLANG_ENABLE_SPLIT_DEBUG_INFO=OFF"
        }
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" ..
                         (package:is_debug() and "Debug" or "Release"))

        import("package.tools.cmake").build(package, configs,
                                            {targets = {"slang"}})
        local installdir = package:installdir()
        local builddir = package:builddir()
        os.cp(path.join(builddir, "Release/*"), installdir)
        os.cp(path.join(builddir, "external/cmark/src/libcmark-gfm.*"),
              path.join(installdir, "lib"))
    end)

    on_test(function(package)
        assert(package:check_cxxsnippets({
            test = [[
            #include <slang-com-ptr.h>
            #include <slang.h>

            void test() {
                Slang::ComPtr<slang::IGlobalSession> global_session;
                slang::createGlobalSession(global_session.writeRef());
            }
        ]]
        }, {configs = {languages = "c++17"}}))
    end)
end)
