#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "ggml-backend.h"

#if defined(VOICECLI_MULTI_ISA) && (defined(__x86_64__) || defined(_M_X64))

extern ggml_backend_reg_t voicecli_avx512_ggml_backend_cpu_reg(void);
extern ggml_backend_reg_t voicecli_avx2_ggml_backend_cpu_reg(void);
extern ggml_backend_reg_t voicecli_sse42_ggml_backend_cpu_reg(void);
extern ggml_backend_reg_t voicecli_x64_ggml_backend_cpu_reg(void);

typedef enum {
    VOICECLI_ISA_X64    = 0,
    VOICECLI_ISA_SSE42  = 1,
    VOICECLI_ISA_AVX2   = 2,
    VOICECLI_ISA_AVX512 = 3,
} voicecli_isa_tier_t;

static void cpuid_x86(uint32_t leaf, uint32_t subleaf, uint32_t out[4]) {
    uint32_t a = 0, b = 0, c = 0, d = 0;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
        : "a"(leaf), "c"(subleaf)
    );
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
}

static uint64_t xgetbv_x86(uint32_t index) {
    uint32_t eax = 0, edx = 0;
    __asm__ __volatile__(
        ".byte 0x0f, 0x01, 0xd0" /* xgetbv */
        : "=a"(eax), "=d"(edx)
        : "c"(index)
    );
    return ((uint64_t)edx << 32) | (uint64_t)eax;
}

static int str_eq_nocase(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static voicecli_isa_tier_t detect_best_isa_tier(void) {
    static int initialized = 0;
    static voicecli_isa_tier_t cached_tier = VOICECLI_ISA_X64;

    if (initialized) {
        return cached_tier;
    }

    uint32_t regs[4] = {0, 0, 0, 0};
    cpuid_x86(0, 0, regs);
    uint32_t max_leaf = regs[0];

    int can_sse42 = 0;
    int can_avx2 = 0;
    int can_avx512 = 0;

    if (max_leaf >= 1) {
        cpuid_x86(1, 0, regs);
        uint32_t ecx = regs[2];

        int sse3    = (ecx >> 0)  & 1;
        int ssse3   = (ecx >> 9)  & 1;
        int fma     = (ecx >> 12) & 1;
        int sse41   = (ecx >> 19) & 1;
        int sse42   = (ecx >> 20) & 1;
        int popcnt  = (ecx >> 23) & 1;
        int xsave   = (ecx >> 26) & 1;
        int osxsave = (ecx >> 27) & 1;
        int avx     = (ecx >> 28) & 1;
        int f16c    = (ecx >> 29) & 1;

        can_sse42 = sse3 && ssse3 && sse41 && sse42 && popcnt;

        int os_avx = 0;
        int os_avx512 = 0;
        if (xsave && osxsave) {
            uint64_t xcr0 = xgetbv_x86(0);
            os_avx    = ((xcr0 & 0x06ULL) == 0x06ULL);
            os_avx512 = ((xcr0 & 0xe6ULL) == 0xe6ULL);
        }

        if (max_leaf >= 7 && can_sse42 && os_avx && avx && f16c && fma) {
            cpuid_x86(7, 0, regs);
            uint32_t ebx = regs[1];

            int bmi1     = (ebx >> 3)  & 1;
            int avx2     = (ebx >> 5)  & 1;
            int bmi2     = (ebx >> 8)  & 1;
            int avx512f  = (ebx >> 16) & 1;
            int avx512dq = (ebx >> 17) & 1;
            int avx512cd = (ebx >> 28) & 1;
            int avx512bw = (ebx >> 30) & 1;
            int avx512vl = (ebx >> 31) & 1;

            can_avx2 = bmi1 && bmi2 && avx2;
            can_avx512 = can_avx2 && os_avx512 &&
                         avx512f && avx512dq && avx512cd && avx512bw && avx512vl;
        }
    }

    voicecli_isa_tier_t hw_best = VOICECLI_ISA_X64;
    if (can_avx512) {
        hw_best = VOICECLI_ISA_AVX512;
    } else if (can_avx2) {
        hw_best = VOICECLI_ISA_AVX2;
    } else if (can_sse42) {
        hw_best = VOICECLI_ISA_SSE42;
    }

    voicecli_isa_tier_t selected = hw_best;
    const char *env_simd = getenv("VOICECLI_SIMD");
    if (env_simd && env_simd[0] != '\0' && !str_eq_nocase(env_simd, "auto")) {
        voicecli_isa_tier_t req = hw_best;
        if (str_eq_nocase(env_simd, "x64") ||
            str_eq_nocase(env_simd, "sse2") ||
            str_eq_nocase(env_simd, "generic") ||
            str_eq_nocase(env_simd, "baseline")) {
            req = VOICECLI_ISA_X64;
        } else if (str_eq_nocase(env_simd, "sse42") || str_eq_nocase(env_simd, "sse4.2")) {
            req = VOICECLI_ISA_SSE42;
        } else if (str_eq_nocase(env_simd, "avx2")) {
            req = VOICECLI_ISA_AVX2;
        } else if (str_eq_nocase(env_simd, "avx512") || str_eq_nocase(env_simd, "avx-512")) {
            req = VOICECLI_ISA_AVX512;
        }
        /* Clamp requested tier to what the host CPU + OS actually support */
        selected = (req <= hw_best) ? req : hw_best;
    }

    cached_tier = selected;
    initialized = 1;
    return cached_tier;
}

const char *voicecli_get_active_simd_name(void) {
    switch (detect_best_isa_tier()) {
        case VOICECLI_ISA_AVX512:
            return "AVX-512 (x86-64-v4)";
        case VOICECLI_ISA_AVX2:
            return "AVX2 + FMA + F16C (x86-64-v3)";
        case VOICECLI_ISA_SSE42:
            return "SSE4.2 (x86-64-v2)";
        case VOICECLI_ISA_X64:
        default:
            return "Baseline x86-64 (SSE2)";
    }
}

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    switch (detect_best_isa_tier()) {
        case VOICECLI_ISA_AVX512:
            return voicecli_avx512_ggml_backend_cpu_reg();
        case VOICECLI_ISA_AVX2:
            return voicecli_avx2_ggml_backend_cpu_reg();
        case VOICECLI_ISA_SSE42:
            return voicecli_sse42_ggml_backend_cpu_reg();
        case VOICECLI_ISA_X64:
        default:
            return voicecli_x64_ggml_backend_cpu_reg();
    }
}

#else

const char *voicecli_get_active_simd_name(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64 NEON";
#else
    return "Native SIMD";
#endif
}

#endif
