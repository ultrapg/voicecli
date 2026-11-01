#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

namespace qwen3_tts {

enum class backend_preference {
    auto_select = 0,
    cpu = 1,
    cuda = 2,
};

// Cross-platform read-only memory-mapped file for zero-copy GGUF tensor loading
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;

    bool open(const std::string & path, std::string & error_msg);
    void close();

    const uint8_t * data() const { return static_cast<const uint8_t *>(data_); }
    uint8_t * mutable_data() const { return static_cast<uint8_t *>(data_); }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr && size_ > 0; }

    // Hint to OS that clean file-backed pages in the interior of [offset, offset + len)
    // have been copied/repacked elsewhere and do not need to stay resident in RSS.
    void advise_dontneed(size_t offset, size_t len) const;

private:
    void * data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    void * file_handle_ = nullptr;
    void * map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// Generic GGUF model loader class
// This is a simplified loader that can be extended for specific model types
class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();
    
    // Open GGUF file and parse metadata
    bool open(const std::string & path);
    
    // Close file and free resources
    void close();
    
    // Get error message if operation failed
    const std::string & get_error() const { return error_msg_; }
    
    // Get number of tensors in file
    int64_t get_n_tensors() const;
    
    // Get tensor name by index
    const char * get_tensor_name(int64_t idx) const;
    
    // Get tensor type by index
    enum ggml_type get_tensor_type(int64_t idx) const;
    
    // Get tensor offset by index
    size_t get_tensor_offset(int64_t idx) const;
    
    // Get tensor size by index
    size_t get_tensor_size(int64_t idx) const;
    
    // Get metadata value (returns -1 if not found)
    int32_t get_u32(const char * key, int32_t default_val = 0) const;
    float get_f32(const char * key, float default_val = 0.0f) const;
    
    // Get data offset (start of tensor data in file)
    size_t get_data_offset() const;
    
    // Get GGUF context (for advanced usage)
    struct gguf_context * get_ctx() const { return ctx_; }
    
    // Get metadata context
    struct ggml_context * get_meta_ctx() const { return meta_ctx_; }
    
protected:
    struct gguf_context * ctx_ = nullptr;
    struct ggml_context * meta_ctx_ = nullptr;
    std::string error_msg_;
    std::string file_path_;
};

// Helper function to allocate and load tensor data from GGUF file.
// When out_mapped_file and out_mmap_buffer are provided on a CPU backend, matching-type
// tensors are mapped zero-copy directly from the GGUF file without heap allocation.
bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type = GGML_BACKEND_DEVICE_TYPE_CPU,
    std::shared_ptr<MappedFile> * out_mapped_file = nullptr,
    ggml_backend_buffer_t * out_mmap_buffer = nullptr
);

// Convert an arbitrary 1D/contiguous row of GGUF source weights to F32 or F16
bool convert_tensor_bytes_to_f32(enum ggml_type src_type, const void * src, float * dst, int64_t n_elems);
bool convert_tensor_bytes_to_f16(enum ggml_type src_type, const void * src, ggml_fp16_t * dst, int64_t n_elems);

// CPU weight repacking helpers (via dynamic backend registry lookup, multi-ISA safe)
ggml_backend_buffer_type_t get_cpu_repack_buffer_type(ggml_backend_t backend);
bool can_repack_mul_mat_weight(ggml_backend_dev_t dev,
                               ggml_backend_buffer_type_t repack_buft,
                               enum ggml_type type,
                               int64_t ne0,
                               int64_t ne1);

// Helper to initialize backend with GPU preference and CPU fallback
ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg);
ggml_backend_t init_cpu_backend(const char * component_name, std::string * error_msg);
void release_preferred_backend(ggml_backend_t backend);
bool set_backend_preference(backend_preference preference);
backend_preference get_backend_preference();
int32_t default_cpu_thread_count();
int32_t default_parallel_thread_count();
bool set_cpu_thread_count(int32_t n_threads, bool explicit_value = true);
int32_t get_cpu_thread_count();
bool cpu_thread_count_is_explicit();
enum ggml_backend_dev_type get_preferred_backend_type();
int32_t get_compiled_backend_mask();
std::string get_active_backend_name();

// Helper function to free model resources
void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer);

} // namespace qwen3_tts
