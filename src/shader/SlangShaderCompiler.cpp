#include "SlangShaderCompiler.hpp"

#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

using Slang::ComPtr;

namespace {

constexpr uint32_t SPIRV_MAGIC = 0x07230203;

std::string getDiagnostics(slang::IBlob* diagnostics) {
    if (diagnostics == nullptr || diagnostics->getBufferSize() == 0) {
        return {};
    }

    const auto* data =
        static_cast<const char*>(diagnostics->getBufferPointer());
    size_t size = diagnostics->getBufferSize();
    if (data[size - 1] == '\0') {
        --size;
    }
    return std::string(data, size);
}

void printDiagnostics(const std::string& diagnostics) {
    if (diagnostics.empty()) {
        return;
    }

    std::cerr << diagnostics;
    if (diagnostics.back() != '\n') {
        std::cerr << '\n';
    }
}

void checkSlangResult(
    SlangResult result,
    slang::IBlob* diagnostics,
    const char* operation
) {
    std::string message = getDiagnostics(diagnostics);

    if (SLANG_FAILED(result)) {
        std::string error = std::string("Slang operation failed: ") + operation;
        if (!message.empty()) {
            error += '\n';
            error += message;
        }
        throw std::runtime_error(error);
    }

    printDiagnostics(message);
}

void checkSlangPointer(
    const void* pointer,
    slang::IBlob* diagnostics,
    const char* operation
) {
    std::string message = getDiagnostics(diagnostics);
    if (pointer == nullptr) {
        std::string error = std::string("Slang operation failed: ") + operation;
        if (!message.empty()) {
            error += '\n';
            error += message;
        }
        throw std::runtime_error(error);
    }

    printDiagnostics(message);
}

}  // namespace

struct SlangShaderCompiler::Impl {
    ComPtr<slang::IGlobalSession> globalSession;
    ComPtr<slang::ISession> session;
    std::mutex mutex;
    uint64_t nextModuleId = 0;

    Impl() {
        checkSlangResult(
            slang::createGlobalSession(globalSession.writeRef()),
            nullptr,
            "createGlobalSession"
        );
        std::array options = {
            slang::CompilerOptionEntry{
                .name = slang::CompilerOptionName::Optimization,
                .value = {
                    .kind = slang::CompilerOptionValueKind::Int,
                    .intValue0 = SLANG_OPTIMIZATION_LEVEL_MAXIMAL
                }
            },
            slang::CompilerOptionEntry{
                .name = slang::CompilerOptionName::Capability,
                .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvCooperativeMatrixKHR"},
            },
            slang::CompilerOptionEntry{
                .name = slang::CompilerOptionName::Capability,
                .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvGroupNonUniform"},
            },
        };
        slang::TargetDesc targetDesc = {
            .format = SLANG_SPIRV,
            .profile = globalSession->findProfile("spirv_1_6"),
            .compilerOptionEntries = options.data(),
            .compilerOptionEntryCount =
                static_cast<uint32_t>(options.size()),
        };
        slang::SessionDesc sessionDesc = {
            .targets = &targetDesc,
            .targetCount = 1,
        };
        checkSlangResult(
            globalSession->createSession(
                sessionDesc,
                session.writeRef()
            ),
            nullptr,
            "createSession"
        );
    }
};

SlangShaderCompiler::SlangShaderCompiler()
    : pimpl(std::make_unique<Impl>()) {}

SlangShaderCompiler::~SlangShaderCompiler() = default;

std::vector<uint32_t> SlangShaderCompiler::genSpirv(
    std::string_view source
) {
    if (source.empty()) {
        throw std::invalid_argument("Slang shader source must not be empty");
    }
    if (source.find('\0') != std::string_view::npos) {
        throw std::invalid_argument(
            "Slang shader source must not contain embedded null characters"
        );
    }

    // A session caches modules by name, so each independent source string needs
    // a distinct name. Serializing access also keeps this compiler safe to call
    // from multiple host threads.
    std::scoped_lock lock(pimpl->mutex);
    std::string moduleName =
        "runtime_shader_" + std::to_string(pimpl->nextModuleId++);
    std::string nullTerminatedSource(source);

    ComPtr<slang::IBlob> loadDiagnostics;
    ComPtr<slang::IModule> module(
        pimpl->session->loadModuleFromSourceString(
            moduleName.c_str(),
            nullptr,
            nullTerminatedSource.c_str(),
            loadDiagnostics.writeRef()
        )
    );
    checkSlangPointer(
        module.get(),
        loadDiagnostics.get(),
        "loadModuleFromSourceString"
    );

    ComPtr<slang::IComponentType> linkedProgram;
    ComPtr<slang::IBlob> linkDiagnostics;
    checkSlangResult(
        module->link(linkedProgram.writeRef(), linkDiagnostics.writeRef()),
        linkDiagnostics.get(),
        "link"
    );
    checkSlangPointer(linkedProgram.get(), nullptr, "link");

    ComPtr<slang::IBlob> spirvCode;
    ComPtr<slang::IBlob> codeDiagnostics;
    checkSlangResult(
        linkedProgram->getTargetCode(
            0,
            spirvCode.writeRef(),
            codeDiagnostics.writeRef()
        ),
        codeDiagnostics.get(),
        "getTargetCode"
    );
    checkSlangPointer(spirvCode.get(), nullptr, "getTargetCode");

    const size_t byteSize = spirvCode->getBufferSize();
    if (byteSize == 0 || byteSize % sizeof(uint32_t) != 0) {
        throw std::runtime_error(
            "Slang returned an invalid SPIR-V byte size"
        );
    }

    std::vector<uint32_t> spirv(byteSize / sizeof(uint32_t));
    std::memcpy(
        spirv.data(),
        spirvCode->getBufferPointer(),
        byteSize
    );
    if (spirv.front() != SPIRV_MAGIC) {
        throw std::runtime_error(
            "Slang output does not contain a valid SPIR-V header"
        );
    }

    return spirv;
}
