#pragma once

#include "lru_cache.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fmt/args.h>
#include <fmt/ranges.h>
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace aiter{

#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

static std::once_flag init_libs_lru_cache, init_func_names_lru_cache, init_root_dir_flag;

template<typename K, typename V>
__inline__ void init_lru_cache(std::unique_ptr<LRUCache<K, V>>& lru_cache){
    auto AITER_MAX_CACHE_SIZE = getenv("AITER_MAX_CACHE_SIZE");
    if(!AITER_MAX_CACHE_SIZE){
        AITER_MAX_CACHE_SIZE = "-1";
    }
    int aiter_max_cache_size = atoi(AITER_MAX_CACHE_SIZE);
    lru_cache = std::make_unique<LRUCache<K, V>>(aiter_max_cache_size);
}

static std::filesystem::path aiter_root_dir;

__inline__ void init_root_dir(){
    const char* root = std::getenv("AITER_ROOT_DIR");
    if(root)
    {
        aiter_root_dir = std::filesystem::absolute(root);
    }
    else
    {
        const char* home = std::getenv("HOME");
        if(!home)
            throw std::runtime_error("HOME or AITER_ROOT_DIR must be set");
        aiter_root_dir = std::filesystem::path(home) / ".aiter";
    }
}

__inline__ std::filesystem::path get_root_dir(){
    std::call_once(init_root_dir_flag, init_root_dir);
    return aiter_root_dir;
}

// Keep this layout and compatible-superset selection in sync with utils.py.
__inline__ std::set<std::string> template_archs()
{
    const char* env = std::getenv("GPU_ARCHS");
    std::stringstream input(env ? env : "native");
    std::set<std::string> archs;
    std::string arch;
    while(std::getline(input, arch, ';'))
    {
        arch.erase(std::remove_if(
                       arch.begin(), arch.end(), [](unsigned char c) { return std::isspace(c); }),
                   arch.end());
        arch = arch.substr(0, arch.find(':'));
        if(arch == "native")
        {
            int device;
            hipDeviceProp_t props;
            if(hipGetDevice(&device) != hipSuccess ||
               hipGetDeviceProperties(&props, device) != hipSuccess)
                throw std::runtime_error("Cannot resolve template execution architecture");
            arch = std::string(props.gcnArchName);
            arch = arch.substr(0, arch.find(':'));
        }
        if(arch.empty())
            throw std::runtime_error("Empty template architecture");
        archs.insert(arch);
    }
    if(archs.empty())
        throw std::runtime_error("No template architectures");
    return archs;
}

__inline__ std::filesystem::path template_library_path(const std::string& folder)
{
    auto archs = template_archs();
    std::string key;
    for(const auto& arch : archs)
        key += (key.empty() ? "" : "+") + arch;
    auto root  = get_root_dir() / "build" / "targets";
    auto exact = root / key / folder / "lib.so";
    if(std::filesystem::is_regular_file(exact))
        return exact;
    if(std::filesystem::is_directory(root))
    {
        std::vector<std::filesystem::path> entries;
        for(const auto& entry : std::filesystem::directory_iterator(root))
            entries.push_back(entry.path());
        std::sort(entries.begin(), entries.end());
        for(const auto& entry : entries)
        {
            std::set<std::string> coverage;
            std::stringstream input(entry.filename().string());
            std::string arch;
            while(std::getline(input, arch, '+'))
                coverage.insert(arch);
            auto candidate = entry / folder / "lib.so";
            if(std::includes(coverage.begin(), coverage.end(), archs.begin(), archs.end()) &&
               std::filesystem::is_regular_file(candidate))
                return candidate;
        }
    }
    return exact;
}

__inline__ const std::pair<std::string, int> execute_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    int exitCode;

    FILE* pipe = popen(cmd.c_str(), "r");

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    try {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }

    exitCode = pclose(pipe);
    return {result, exitCode};
}


__inline__ const std::pair<std::string, int> execute_cmd(const std::string& cmd, const std::list<std::string>& args) {
    fmt::dynamic_format_arg_store<fmt::format_context> store;
    for (const auto& arg : args) {
        store.push_back(arg);
    }
    std::string cmd_with_args = fmt::vformat(cmd, store);
    AITER_LOG_INFO(cmd_with_args);
    const auto results = execute_cmd(cmd_with_args);
    AITER_LOG_INFO(results.first);
    return results;
}


class SharedLibrary {
private:
    void* handle;

public:
    SharedLibrary(std::string& path) {
        handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error(dlerror());
        }
    }

    ~SharedLibrary() {
        if (handle) {
            dlclose(handle);
        }
    }

    // Get raw function pointer
    void* getRawFunction(const char* funcName) {
        dlerror(); // Clear any existing error
        void* funcPtr = dlsym(handle, funcName);
        const char* error = dlerror();
        if (error) {
            throw std::runtime_error(error);
        }
        return funcPtr;
    }

    // Template to call function with any return type and arguments
    template<typename ReturnType = void, typename... Args>
    ReturnType call(std::string& func_name, Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction(func_name.c_str()));
        return func(std::forward<Args>(args)...);
    }
};

static std::unique_ptr<LRUCache<std::string, std::shared_ptr<SharedLibrary>>> libs;
static std::unique_ptr<LRUCache<std::string, std::string>> func_names;

template<typename... Args>
__inline__ void run_lib(std::string func_name, std::string folder, Args... args) {
    std::call_once(init_libs_lru_cache, init_lru_cache<std::string, std::shared_ptr<SharedLibrary>>, libs);
    std::string lib_path = template_library_path(folder).string();
    auto func_lib        = libs->get(lib_path);
    if(!func_lib){
        libs->put(lib_path, std::make_shared<SharedLibrary>(lib_path));
        func_lib = libs->get(lib_path);
    }
    (*func_lib)->call(func_name, std::forward<Args>(args)...);
}


__inline__ std::string hash_signature(const std::string& signature) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, signature.data(), signature.size());
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    std::stringstream ss;
    for (unsigned int i = 0; i < digest_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    return ss.str();
}


__inline__ std::string get_default_func_name(const std::string& md_name, std::list<std::string>& args) {
    std::call_once(init_func_names_lru_cache, init_lru_cache<std::string, std::string>, func_names);
    std::string args_str = fmt::format("{}", fmt::join(args, "_"));
    std::transform(args_str.begin(), args_str.end(), args_str.begin(),
    [](unsigned char c){ return std::tolower(c); });
    auto func_name = func_names->get(args_str);
    if(!func_name){
        func_names->put(args_str, fmt::format("{}_{}", md_name, hash_signature(args_str)));
        func_name = func_names->get(args_str);
    }
    return *func_name;
}


__inline__ bool not_built(const std::string& folder)
{ return !std::filesystem::is_regular_file(template_library_path(folder)); }
}
