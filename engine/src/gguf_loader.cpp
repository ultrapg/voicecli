#include "gguf_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sched.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace qwen3_tts {

MappedFile::~MappedFile() {
    close();
}

bool MappedFile::open(const std::string & path, std::string & error_msg) {
    close();

#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        error_msg = "Failed to open file for mmap: " + path;
        return false;
    }
    LARGE_INTEGER file_sz = {};
    if (!GetFileSizeEx(fh, &file_sz) || file_sz.QuadPart <= 0) {
        CloseHandle(fh);
        error_msg = "Failed to get file size for mmap: " + path;
        return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        error_msg = "Failed to create file mapping: " + path;
        return false;
    }
    void * view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mh);
        CloseHandle(fh);
        error_msg = "Failed to map view of file: " + path;
        return false;
    }
    file_handle_ = fh;
    map_handle_ = mh;
    data_ = view;
    size_ = static_cast<size_t>(file_sz.QuadPart);
    return true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg = "Failed to open file for mmap: " + path;
        return false;
    }
    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        error_msg = "Failed to stat file for mmap: " + path;
        return false;
    }
    const size_t map_sz = static_cast<size_t>(st.st_size);
    void * mapped = ::mmap(nullptr, map_sz, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        error_msg = "Failed to mmap file: " + path;
        return false;
    }
    data_ = mapped;
    size_ = map_sz;
    return true;
#endif
}

void MappedFile::close() {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(static_cast<HANDLE>(map_handle_));
        map_handle_ = nullptr;
    }
    if (file_handle_) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
    size_ = 0;
#else
    if (data_ && size_ > 0) {
        ::munmap(data_, size_);
    }
    data_ = nullptr;
    size_ = 0;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

void MappedFile::advise_dontneed(size_t offset, size_t len) const {
#if !defined(_WIN32) && defined(MADV_DONTNEED)
    if (!data_ || size_ == 0 || len == 0 || offset >= size_) {
        return;
    }
    static const size_t page_size = []() -> size_t {
        const long ps = sysconf(_SC_PAGESIZE);
        return ps > 0 ? static_cast<size_t>(ps) : 4096;
    }();
    const size_t clamped_end = std::min(size_, offset + len);
    const size_t start_page = (offset + page_size - 1) & ~(page_size - 1);
    const size_t end_page = clamped_end & ~(page_size - 1);
    if (end_page > start_page) {
        ::madvise(static_cast<uint8_t *>(data_) + start_page,
                  end_page - start_page,
                  MADV_DONTNEED);
    }
#else
    (void) offset;
    (void) len;
#endif
}

namespace {

using ggml_threadpool_new_fn_t = struct ggml_threadpool * (*)(struct ggml_threadpool_params *);
using ggml_threadpool_free_fn_t = void (*)(struct ggml_threadpool *);
using ggml_backend_cpu_set_threadpool_fn_t = void (*)(ggml_backend_t, struct ggml_threadpool *);

struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
    backend_preference preference = backend_preference::auto_select;
    int32_t cpu_n_threads = 0;
    bool cpu_n_threads_explicit = false;
    std::string active_backend_name;

    struct ggml_threadpool * threadpool = nullptr;
    ggml_threadpool_free_fn_t threadpool_free_fn = nullptr;
    int32_t threadpool_n_threads = 0;
    std::vector<ggml_backend_t> live_cpu_backends;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}

int32_t detect_physical_cpu_threads() {
    int32_t detected = 0;

#if defined(__linux__)
    auto read_int_file = [](const char * p, long long & out) -> bool {
        FILE * f = fopen(p, "r");
        if (!f) return false;
        const int rc = fscanf(f, "%lld", &out);
        fclose(f);
        return rc == 1;
    };

    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        std::set<std::pair<int, int>> physical_cores;
        int allowed_logical = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &mask)) continue;
            ++allowed_logical;
            char core_path[128];
            char pkg_path[128];
            snprintf(core_path, sizeof(core_path),
                     "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
            snprintf(pkg_path, sizeof(pkg_path),
                     "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
            long long core_id = -1;
            long long pkg_id = 0;
            if (read_int_file(core_path, core_id) && core_id >= 0) {
                (void) read_int_file(pkg_path, pkg_id);
                physical_cores.emplace(static_cast<int>(pkg_id), static_cast<int>(core_id));
            }
        }
        if (!physical_cores.empty()) {
            detected = static_cast<int32_t>(physical_cores.size());
        } else if (allowed_logical > 0) {
            detected = std::max<int32_t>(1, allowed_logical / 2);
        }
    }

    // Respect Linux container CPU quotas (cgroup v2 & v1) to avoid CFS throttling
    {
        FILE * f = fopen("/sys/fs/cgroup/cpu.max", "r");
        if (f) {
            char quota_str[64] = {};
            long long period = 0;
            if (fscanf(f, "%63s %lld", quota_str, &period) == 2 && period > 0 &&
                strcmp(quota_str, "max") != 0) {
                const long long quota = atoll(quota_str);
                if (quota > 0) {
                    const int32_t quota_cores =
                        static_cast<int32_t>((quota + period - 1) / period);
                    if (quota_cores >= 1 && (detected <= 0 || quota_cores < detected)) {
                        detected = quota_cores;
                    }
                }
            }
            fclose(f);
        } else {
            long long quota_us = -1;
            long long period_us = -1;
            if (read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", quota_us) &&
                read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", period_us) &&
                quota_us > 0 && period_us > 0) {
                const int32_t quota_cores =
                    static_cast<int32_t>((quota_us + period_us - 1) / period_us);
                if (quota_cores >= 1 && (detected <= 0 || quota_cores < detected)) {
                    detected = quota_cores;
                }
            }
        }
    }
#elif defined(__APPLE__)
    int32_t p_cores = 0;
    size_t sz = sizeof(p_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &p_cores, &sz, nullptr, 0) == 0 && p_cores > 0) {
        detected = p_cores;
    } else {
        sz = sizeof(p_cores);
        if (sysctlbyname("hw.physicalcpu", &p_cores, &sz, nullptr, 0) == 0 && p_cores > 0) {
            detected = p_cores;
        }
    }
#endif

    if (detected <= 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        detected = hw > 1 ? static_cast<int32_t>(hw / 2) : 1;
    }

    // Cap automatic default at 16 threads so multi-socket / many-core servers do not
    // spend more time in per-op GEMV barriers than memory bandwidth allows.
    return std::clamp<int32_t>(detected, 1, 16);
}

int32_t sanitize_thread_count(int32_t n_threads) {
    return n_threads > 0 ? n_threads : default_cpu_thread_count();
}

void detach_cpu_backend_threadpool(ggml_backend_t backend) {
    if (!backend) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (!reg) return;
    auto set_tp = reinterpret_cast<ggml_backend_cpu_set_threadpool_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    if (set_tp) {
        set_tp(backend, nullptr);
    }
}

void sync_all_cpu_backends_threads() {
    auto & shared = get_shared_backend_state();
    const int32_t n_threads = get_cpu_thread_count();

    if (shared.live_cpu_backends.empty()) {
        if (shared.threadpool && shared.threadpool_n_threads != n_threads) {
            if (shared.threadpool_free_fn) {
                shared.threadpool_free_fn(shared.threadpool);
            }
            shared.threadpool = nullptr;
            shared.threadpool_n_threads = 0;
        }
        return;
    }

    ggml_backend_dev_t dev0 = ggml_backend_get_device(shared.live_cpu_backends.front());
    ggml_backend_reg_t reg = dev0 ? ggml_backend_dev_backend_reg(dev0) : nullptr;
    if (!reg) return;

    auto set_n_fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    auto set_tp_fn = reinterpret_cast<ggml_backend_cpu_set_threadpool_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    auto tp_new_fn = reinterpret_cast<ggml_threadpool_new_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    auto tp_free_fn = reinterpret_cast<ggml_threadpool_free_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));

    const bool need_pool = (n_threads > 1 && tp_new_fn != nullptr && set_tp_fn != nullptr);
    const bool pool_mismatch =
        (!need_pool && shared.threadpool != nullptr) ||
        (need_pool && (shared.threadpool == nullptr || shared.threadpool_n_threads != n_threads));

    if (pool_mismatch) {
        if (set_tp_fn) {
            for (ggml_backend_t b : shared.live_cpu_backends) {
                set_tp_fn(b, nullptr);
            }
        }
        if (shared.threadpool && shared.threadpool_free_fn) {
            shared.threadpool_free_fn(shared.threadpool);
        }
        shared.threadpool = nullptr;
        shared.threadpool_n_threads = 0;

        if (need_pool) {
            struct ggml_threadpool_params tp_params = ggml_threadpool_params_default(n_threads);
            shared.threadpool = tp_new_fn(&tp_params);
            shared.threadpool_free_fn = tp_free_fn;
            shared.threadpool_n_threads = shared.threadpool ? n_threads : 0;
        }
    }

    for (ggml_backend_t b : shared.live_cpu_backends) {
        if (set_n_fn) {
            set_n_fn(b, n_threads);
        }
        if (set_tp_fn) {
            set_tp_fn(b, shared.threadpool);
        }
    }
}

void register_cpu_backend_and_configure(ggml_backend_t backend) {
    if (!backend) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    auto & shared = get_shared_backend_state();
    if (std::find(shared.live_cpu_backends.begin(), shared.live_cpu_backends.end(), backend) ==
        shared.live_cpu_backends.end()) {
        shared.live_cpu_backends.push_back(backend);
    }
    sync_all_cpu_backends_threads();
}

void unregister_and_free_backend(ggml_backend_t backend) {
    if (!backend) return;
    auto & shared = get_shared_backend_state();
    auto it = std::find(shared.live_cpu_backends.begin(), shared.live_cpu_backends.end(), backend);
    if (it != shared.live_cpu_backends.end()) {
        detach_cpu_backend_threadpool(backend);
        shared.live_cpu_backends.erase(it);
    }
    ggml_backend_free(backend);
}

ggml_backend_t init_backend_by_type(enum ggml_backend_dev_type type) {
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (backend && type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        register_cpu_backend_and_configure(backend);
    }
    return backend;
}

std::string backend_name(ggml_backend_t backend) {
    if (!backend) {
        return "";
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    return name ? name : "Unknown";
}

} // namespace

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    auto & shared = get_shared_backend_state();
    if (shared.backend) {
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = nullptr;
    if (shared.preference == backend_preference::cpu) {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else if (shared.preference == backend_preference::cuda) {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    } else {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        }
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
        }
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        if (shared.preference == backend_preference::cuda) {
            *error_msg = "CUDA backend was requested but could not be initialized for " + std::string(name);
        } else if (shared.preference == backend_preference::cpu) {
            *error_msg = "CPU backend could not be initialized for " + std::string(name);
        } else {
            *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
        }
    }

    if (backend) {
        shared.backend = backend;
        shared.ref_count = 1;
        shared.active_backend_name = backend_name(backend);
        const char * dbg = std::getenv("VOICECLI_DEBUG");
        if (dbg && dbg[0] != '\0' && dbg[0] != '0') {
            fprintf(stderr, "  Native backend preference: %d\n", static_cast<int>(shared.preference));
            fprintf(stderr, "  Native active backend: %s\n", shared.active_backend_name.c_str());
            if (ggml_backend_get_device(backend) &&
                ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                fprintf(stderr, "  Native CPU threads: %d\n", get_cpu_thread_count());
            }
        }
    }

    return backend;
}

ggml_backend_t init_cpu_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    ggml_backend_t backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize CPU backend for " + std::string(name);
    }
    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    auto & shared = get_shared_backend_state();
    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            unregister_and_free_backend(shared.backend);
            shared.backend = nullptr;
            shared.ref_count = 0;
            shared.active_backend_name.clear();
        }
        return;
    }

    unregister_and_free_backend(backend);
}

bool set_backend_preference(backend_preference preference) {
    auto & shared = get_shared_backend_state();
    if (shared.backend && shared.preference != preference) {
        return false;
    }
    shared.preference = preference;
    return true;
}

int32_t default_cpu_thread_count() {
    static const int32_t cached_threads = detect_physical_cpu_threads();
    return cached_threads;
}

int32_t default_parallel_thread_count() {
    int32_t n_threads = (int32_t) std::thread::hardware_concurrency();
    return n_threads > 0 ? n_threads : default_cpu_thread_count();
}

bool set_cpu_thread_count(int32_t n_threads, bool explicit_value) {
    auto & shared = get_shared_backend_state();
    shared.cpu_n_threads = sanitize_thread_count(n_threads);
    shared.cpu_n_threads_explicit = explicit_value;
    sync_all_cpu_backends_threads();
    return true;
}

int32_t get_cpu_thread_count() {
    auto & shared = get_shared_backend_state();
    if (shared.cpu_n_threads <= 0) {
        shared.cpu_n_threads = default_cpu_thread_count();
    }
    return shared.cpu_n_threads;
}

bool cpu_thread_count_is_explicit() {
    return get_shared_backend_state().cpu_n_threads_explicit;
}

backend_preference get_backend_preference() {
    return get_shared_backend_state().preference;
}

enum ggml_backend_dev_type get_preferred_backend_type() {
    switch (get_backend_preference()) {
        case backend_preference::cpu:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case backend_preference::cuda:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case backend_preference::auto_select:
        default:
            return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
}

int32_t get_compiled_backend_mask() {
    int32_t mask = 1; // CPU
#ifdef QWEN3_TTS_CUDA_ENABLED
    mask |= 2; // CUDA
#endif
    return mask;
}

std::string get_active_backend_name() {
    return get_shared_backend_state().active_backend_name;
}

bool convert_tensor_bytes_to_f32(enum ggml_type src_type, const void * src, float * dst, int64_t n_elems) {
    if (n_elems <= 0) return true;
    if (!src || !dst) return false;
    if (src_type == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n_elems) * sizeof(float));
        return true;
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(src_type);
    if (!traits || !traits->to_float) {
        return false;
    }
    traits->to_float(src, dst, n_elems);
    return true;
}

bool convert_tensor_bytes_to_f16(enum ggml_type src_type, const void * src, ggml_fp16_t * dst, int64_t n_elems) {
    if (n_elems <= 0) return true;
    if (!src || !dst) return false;
    if (src_type == GGML_TYPE_F16) {
        std::memcpy(dst, src, static_cast<size_t>(n_elems) * sizeof(ggml_fp16_t));
        return true;
    }
    if (src_type == GGML_TYPE_F32) {
        ggml_fp32_to_fp16_row(static_cast<const float *>(src), dst, n_elems);
        return true;
    }
    if (src_type == GGML_TYPE_BF16) {
        const ggml_bf16_t * bf = static_cast<const ggml_bf16_t *>(src);
        constexpr int64_t chunk = 4096;
        float tmp[chunk];
        int64_t done = 0;
        while (done < n_elems) {
            const int64_t cur = std::min<int64_t>(chunk, n_elems - done);
            ggml_bf16_to_fp32_row(bf + done, tmp, cur);
            ggml_fp32_to_fp16_row(tmp, dst + done, cur);
            done += cur;
        }
        return true;
    }
    std::vector<float> tmp(static_cast<size_t>(n_elems));
    if (!convert_tensor_bytes_to_f32(src_type, src, tmp.data(), n_elems)) {
        return false;
    }
    ggml_fp32_to_fp16_row(tmp.data(), dst, n_elems);
    return true;
}

ggml_backend_buffer_type_t get_cpu_repack_buffer_type(ggml_backend_t backend) {
    if (!backend) return nullptr;
    const char * disable_env = std::getenv("VOICECLI_DISABLE_REPACK");
    if (disable_env && disable_env[0] != '\0' && disable_env[0] != '0') {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return nullptr;

    auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    if (!get_extra_bufts) return nullptr;

    ggml_backend_buffer_type_t * bufts = get_extra_bufts(dev);
    if (!bufts) return nullptr;
    for (size_t i = 0; bufts[i] != nullptr; ++i) {
        const char * name = ggml_backend_buft_name(bufts[i]);
        if (name && std::strncmp(name, "CPU_REPACK", 10) == 0) {
            return bufts[i];
        }
    }
    return nullptr;
}

bool can_repack_mul_mat_weight(ggml_backend_dev_t dev,
                               ggml_backend_buffer_type_t repack_buft,
                               enum ggml_type type,
                               int64_t ne0,
                               int64_t ne1) {
    if (!dev || !repack_buft || ne0 <= 0 || ne1 <= 0) {
        return false;
    }
    const int64_t blck = ggml_blck_size(type);
    if (blck <= 0 || (ne0 % blck) != 0) {
        return false;
    }

    static ggml_backend_buffer_type_t s_cached_buft = nullptr;
    static ggml_backend_buffer_t s_dummy_buf = nullptr;
    if (s_cached_buft != repack_buft || !s_dummy_buf) {
        if (s_dummy_buf) {
            ggml_backend_buffer_free(s_dummy_buf);
            s_dummy_buf = nullptr;
        }
        s_dummy_buf = ggml_backend_buft_alloc_buffer(repack_buft, 64);
        s_cached_buft = repack_buft;
    }
    if (!s_dummy_buf) {
        return false;
    }

    uint8_t meta_buf[4096];
    struct ggml_init_params params = {
        /*.mem_size   =*/ sizeof(meta_buf),
        /*.mem_buffer =*/ meta_buf,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * probe_ctx = ggml_init(params);
    if (!probe_ctx) {
        return false;
    }

    struct ggml_tensor * w = ggml_new_tensor_2d(probe_ctx, type, ne0, ne1);
    struct ggml_tensor * x = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, ne0, 1);
    struct ggml_tensor * op = (w && x) ? ggml_mul_mat(probe_ctx, w, x) : nullptr;
    bool supported = false;
    if (op) {
        w->buffer = s_dummy_buf;
        supported = ggml_backend_dev_supports_op(dev, op);
        w->buffer = nullptr;
    }
    ggml_free(probe_ctx);
    return supported;
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type,
    std::shared_ptr<MappedFile> * out_mapped_file,
    ggml_backend_buffer_t * out_mmap_buffer
) {
    buffer = nullptr;
    if (out_mapped_file) *out_mapped_file = nullptr;
    if (out_mmap_buffer) *out_mmap_buffer = nullptr;

    ggml_backend_t backend = nullptr;
    bool release_via_preferred = false;
    if (preferred_backend_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
        get_backend_preference() != backend_preference::cuda) {
        backend = init_preferred_backend("GGUF tensor loader", &error_msg);
        release_via_preferred = (backend != nullptr);
    }
    if (!backend) {
        backend = init_backend_by_type(preferred_backend_type);
        const backend_preference preference = get_backend_preference();
        const bool allow_fallback = preference == backend_preference::auto_select &&
            preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU;
        if (!backend && allow_fallback) {
            if (preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
            }
            if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_GPU) {
                backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            }
            if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
            }
            if (!backend) {
                backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
        }
    }
    if (!backend) {
        const backend_preference preference = get_backend_preference();
        if (preference == backend_preference::cuda) {
            error_msg = "CUDA backend was requested but could not be initialized for GGUF tensor loader";
        } else if (preference == backend_preference::cpu) {
            error_msg = "CPU backend could not be initialized for GGUF tensor loader";
        } else {
            error_msg = "Failed to initialize backend for GGUF tensor loader";
        }
        return false;
    }

    auto cleanup_backend = [&]() {
        if (release_via_preferred) {
            release_preferred_backend(backend);
        } else {
            unregister_and_free_backend(backend);
        }
    };

    auto mapped = std::make_shared<MappedFile>();
    if (!mapped->open(path, error_msg)) {
        cleanup_backend();
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool is_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const bool can_zero_copy_mmap = is_cpu && out_mapped_file != nullptr && out_mmap_buffer != nullptr;

    struct tensor_file_info {
        int64_t gguf_idx = -1;
        enum ggml_type src_type = GGML_TYPE_F32;
        size_t offset = 0;
        size_t src_nbytes = 0;
    };
    std::map<struct ggml_tensor *, tensor_file_info> file_info_by_tensor;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        auto it = tensors.find(name);
        if (it == tensors.end() || !it->second) {
            continue;
        }
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        const size_t src_nbytes = gguf_get_tensor_size(ctx, i);
        if (data_offset + offset + src_nbytes > mapped->size()) {
            error_msg = "GGUF tensor out of file bounds: " + std::string(name);
            cleanup_backend();
            return false;
        }
        file_info_by_tensor[it->second] = {
            i,
            gguf_get_tensor_type(ctx, i),
            offset,
            src_nbytes,
        };
    }

    ggml_backend_buffer_t mmap_buf = nullptr;
    if (can_zero_copy_mmap) {
        mmap_buf = ggml_backend_cpu_buffer_from_ptr(mapped->mutable_data(), mapped->size());
        if (!mmap_buf) {
            error_msg = "Failed to create zero-copy mmap GGML buffer";
            cleanup_backend();
            return false;
        }
        ggml_backend_buffer_set_usage(mmap_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    std::vector<struct ggml_tensor *> alloc_tensors;
    for (struct ggml_tensor * t = ggml_get_first_tensor(model_ctx);
         t != nullptr;
         t = ggml_get_next_tensor(model_ctx, t)) {
        if (t->view_src != nullptr || t->buffer != nullptr) {
            continue;
        }
        auto info_it = file_info_by_tensor.find(t);
        if (can_zero_copy_mmap && info_it != file_info_by_tensor.end()) {
            const auto & info = info_it->second;
            const size_t dst_nbytes = ggml_nbytes(t);
            const uintptr_t ptr_val =
                reinterpret_cast<uintptr_t>(mapped->mutable_data() + data_offset + info.offset);
            if (info.src_type == t->type && info.src_nbytes == dst_nbytes && (ptr_val % 32) == 0) {
                if (ggml_backend_tensor_alloc(
                        mmap_buf, t, mapped->mutable_data() + data_offset + info.offset) != GGML_STATUS_SUCCESS) {
                    error_msg = "Failed to bind zero-copy tensor: " + std::string(ggml_get_name(t));
                    ggml_backend_buffer_free(mmap_buf);
                    cleanup_backend();
                    return false;
                }
                continue;
            }
        }
        alloc_tensors.push_back(t);
    }

    if (!alloc_tensors.empty()) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t total_size = 0;
        for (struct ggml_tensor * t : alloc_tensors) {
            size_t sz = ggml_backend_buft_get_alloc_size(buft, t);
            sz = (sz + align - 1) & ~(align - 1);
            total_size += sz;
        }
        total_size = std::max<size_t>(total_size, align);
        buffer = ggml_backend_buft_alloc_buffer(buft, total_size);
        if (!buffer) {
            error_msg = "Failed to allocate tensor buffer";
            if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
            cleanup_backend();
            return false;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        uint8_t * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
        size_t cur_off = 0;
        for (struct ggml_tensor * t : alloc_tensors) {
            size_t sz = ggml_backend_buft_get_alloc_size(buft, t);
            sz = (sz + align - 1) & ~(align - 1);
            if (ggml_backend_tensor_alloc(buffer, t, base + cur_off) != GGML_STATUS_SUCCESS) {
                error_msg = "Failed to allocate tensor in buffer: " + std::string(ggml_get_name(t));
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
                if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                cleanup_backend();
                return false;
            }
            cur_off += sz;
        }
    }

    // Initialize any view tensors in model_ctx
    for (struct ggml_tensor * t = ggml_get_first_tensor(model_ctx);
         t != nullptr;
         t = ggml_get_next_tensor(model_ctx, t)) {
        if (t->view_src != nullptr && t->buffer == nullptr) {
            if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                error_msg = "Failed to init tensor view: " + std::string(ggml_get_name(t));
                if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                cleanup_backend();
                return false;
            }
        }
    }

    std::vector<ggml_fp16_t> f16_scratch;
    std::vector<float> f32_scratch;

    for (const auto & kv : file_info_by_tensor) {
        struct ggml_tensor * tensor = kv.first;
        const auto & info = kv.second;
        if (mmap_buf && tensor->buffer == mmap_buf) {
            continue; // Already bound zero-copy!
        }

        const uint8_t * src_ptr = mapped->data() + data_offset + info.offset;
        const size_t target_nbytes = ggml_nbytes(tensor);
        const char * name = gguf_get_tensor_name(ctx, info.gguf_idx);

        if (info.src_type == tensor->type) {
            if (info.src_nbytes != target_nbytes) {
                error_msg = "Tensor byte size mismatch: " + std::string(name);
                if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                cleanup_backend();
                return false;
            }
            ggml_backend_tensor_set(tensor, src_ptr, 0, target_nbytes);
        } else if (tensor->type == GGML_TYPE_F16) {
            const int64_t n = ggml_nelements(tensor);
            if (static_cast<size_t>(n) * sizeof(ggml_fp16_t) != target_nbytes) {
                error_msg = "Converted F16 tensor byte size mismatch: " + std::string(name);
                if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                cleanup_backend();
                return false;
            }
            if (ggml_backend_buffer_is_host(tensor->buffer)) {
                if (!convert_tensor_bytes_to_f16(info.src_type, src_ptr,
                                                 static_cast<ggml_fp16_t *>(tensor->data), n)) {
                    error_msg = "Unsupported F16 conversion for " + std::string(name);
                    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                    if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                    cleanup_backend();
                    return false;
                }
            } else {
                f16_scratch.resize(static_cast<size_t>(n));
                if (!convert_tensor_bytes_to_f16(info.src_type, src_ptr, f16_scratch.data(), n)) {
                    error_msg = "Unsupported F16 conversion for " + std::string(name);
                    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                    if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                    cleanup_backend();
                    return false;
                }
                ggml_backend_tensor_set(tensor, f16_scratch.data(), 0, target_nbytes);
            }
        } else if (tensor->type == GGML_TYPE_F32) {
            const int64_t n = ggml_nelements(tensor);
            if (static_cast<size_t>(n) * sizeof(float) != target_nbytes) {
                error_msg = "Converted F32 tensor byte size mismatch: " + std::string(name);
                if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                cleanup_backend();
                return false;
            }
            if (ggml_backend_buffer_is_host(tensor->buffer)) {
                if (!convert_tensor_bytes_to_f32(info.src_type, src_ptr,
                                                 static_cast<float *>(tensor->data), n)) {
                    error_msg = "Unsupported F32 conversion for " + std::string(name);
                    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                    if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                    cleanup_backend();
                    return false;
                }
            } else {
                f32_scratch.resize(static_cast<size_t>(n));
                if (!convert_tensor_bytes_to_f32(info.src_type, src_ptr, f32_scratch.data(), n)) {
                    error_msg = "Unsupported F32 conversion for " + std::string(name);
                    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                    if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
                    cleanup_backend();
                    return false;
                }
                ggml_backend_tensor_set(tensor, f32_scratch.data(), 0, target_nbytes);
            }
        } else {
            error_msg = "Unsupported tensor conversion for " + std::string(name) + ": " +
                        ggml_type_name(info.src_type) + " -> " + ggml_type_name(tensor->type);
            if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (mmap_buf) ggml_backend_buffer_free(mmap_buf);
            cleanup_backend();
            return false;
        }

        // Source pages were copied/converted into a separate buffer; release page cache RSS
        mapped->advise_dontneed(data_offset + info.offset, info.src_nbytes);
    }

    if (can_zero_copy_mmap) {
        *out_mapped_file = mapped;
        *out_mmap_buffer = mmap_buf;
    }

    cleanup_backend();
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
