#ifndef SLANGSHADERCOMPILER_HPP
#define SLANGSHADERCOMPILER_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <string_view>

struct SlangShaderCompiler {
public:
    SlangShaderCompiler();
    ~SlangShaderCompiler();

    // Module, include not implemented yet.
    std::vector<uint32_t> genSpirv(std::string_view source);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

#endif  // SLANGSHADERCOMPILER_HPP
