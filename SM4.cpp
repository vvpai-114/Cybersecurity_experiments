#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <wmmintrin.h>

// SM4 S-box
static const uint8_t SM4_SBOX[256] = {
    0xD6, 0x90, 0xE9, 0xFE, 0xCC, 0xE1, 0x3D, 0xB7, 0x16, 0xB6, 0x14, 0xC2, 0x28, 0xFB, 0x2C, 0x05,
    0x2B, 0x67, 0x9A, 0x76, 0x2A, 0xBE, 0x04, 0xC3, 0xAA, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9C, 0x42, 0x50, 0xF4, 0x91, 0xEF, 0x98, 0x7A, 0x33, 0x54, 0x0B, 0x43, 0xED, 0xCF, 0xAC, 0x62,
    0xE4, 0xB3, 0x1C, 0xA9, 0xC9, 0x08, 0xE8, 0x95, 0x80, 0xDF, 0x94, 0xFA, 0x75, 0x8F, 0x3F, 0xA6,
    0x47, 0x07, 0xA7, 0xFC, 0xF3, 0x73, 0x17, 0xBA, 0x83, 0x59, 0x3C, 0x19, 0xE6, 0x85, 0x4F, 0xA8,
    0x68, 0x6B, 0x81, 0xB2, 0x71, 0x64, 0xDA, 0x8B, 0xF8, 0xEB, 0x0F, 0x4B, 0x70, 0x56, 0x9D, 0x35,
    0x1E, 0x24, 0x0E, 0x5E, 0x63, 0x58, 0xD1, 0xA2, 0x25, 0x22, 0x7C, 0x3B, 0x01, 0x21, 0x78, 0x87,
    0xD4, 0x00, 0x46, 0x57, 0x9F, 0xD3, 0x27, 0x52, 0x4C, 0x36, 0x02, 0xE7, 0xA0, 0xC4, 0xC8, 0x9E,
    0xEA, 0xBF, 0x8A, 0xD2, 0x40, 0xC7, 0x38, 0xB5, 0xA3, 0xF7, 0xF2, 0xCE, 0xF9, 0x61, 0x15, 0xA1,
    0xE0, 0xAE, 0x5D, 0xA4, 0x9B, 0x34, 0x1A, 0x55, 0xAD, 0x93, 0x32, 0x30, 0xF5, 0x8C, 0xB1, 0xE3,
    0x1D, 0xF6, 0xE2, 0x2E, 0x82, 0x66, 0xCA, 0x60, 0xC0, 0x29, 0x23, 0xAB, 0x0D, 0x53, 0x4E, 0x6F,
    0xD5, 0xDB, 0x37, 0x45, 0xDE, 0xFD, 0x8E, 0x2F, 0x03, 0xFF, 0x6A, 0x72, 0x6D, 0x6C, 0x5B, 0x51,
    0x8D, 0x1B, 0xAF, 0x92, 0xBB, 0xDD, 0xBC, 0x7F, 0x11, 0xD9, 0x5C, 0x41, 0x1F, 0x10, 0x5A, 0xD8,
    0x0A, 0xC1, 0x31, 0x88, 0xA5, 0xCD, 0x7B, 0xBD, 0x2D, 0x74, 0xD0, 0x12, 0xB8, 0xE5, 0xB4, 0xB0,
    0x89, 0x69, 0x97, 0x4A, 0x0C, 0x96, 0x77, 0x7E, 0x65, 0xB9, 0xF1, 0x09, 0xC5, 0x6E, 0xC6, 0x84,
    0x18, 0xF0, 0x7D, 0xEC, 0x3A, 0xDC, 0x4D, 0x20, 0x79, 0xEE, 0x5F, 0x3E, 0xD7, 0xCB, 0x39, 0x48
};

// SM4 FK constants
static const uint32_t FK[4] = {
    0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC
};

// SM4 CK constants
static const uint32_t CK[32] = {
    0x00070E15, 0x1C232A31, 0x383F464D, 0x545B6269,
    0x70777E85, 0x8C939AA1, 0xA8AFB6BD, 0xC4CBD2D9,
    0xE0E7EEF5, 0xFC030A11, 0x181F262D, 0x343B4249,
    0x50575E65, 0x6C737A81, 0x888F969D, 0xA4ABB2B9,
    0xC0C7CED5, 0xDCE3EAF1, 0xF8FF060D, 0x141B2229,
    0x30373E45, 0x4C535A61, 0x686F767D, 0x848B9299,
    0xA0A7AEB5, 0xBCC3CAD1, 0xD8DFE6ED, 0xF4FB0209,
    0x10171E25, 0x2C333A41, 0x484F565D, 0x646B7279
};

// Rotate left
static inline uint32_t rotl32(uint32_t x, uint8_t n) {
    return (x << n) | (x >> (32 - n));
}

// ==================== SM4基础实现 ====================
void sm4_key_schedule(const uint8_t key[16], uint32_t rk[32]) {
    uint32_t k[4];

    // 加载密钥
    k[0] = ((uint32_t)key[0] << 24) | ((uint32_t)key[1] << 16) | ((uint32_t)key[2] << 8) | (uint32_t)key[3];
    k[1] = ((uint32_t)key[4] << 24) | ((uint32_t)key[5] << 16) | ((uint32_t)key[6] << 8) | (uint32_t)key[7];
    k[2] = ((uint32_t)key[8] << 24) | ((uint32_t)key[9] << 16) | ((uint32_t)key[10] << 8) | (uint32_t)key[11];
    k[3] = ((uint32_t)key[12] << 24) | ((uint32_t)key[13] << 16) | ((uint32_t)key[14] << 8) | (uint32_t)key[15];

    // 应用FK
    k[0] ^= FK[0];
    k[1] ^= FK[1];
    k[2] ^= FK[2];
    k[3] ^= FK[3];

    // 轮密钥生成
    for (int i = 0; i < 32; i++) {
        uint32_t tmp = k[1] ^ k[2] ^ k[3] ^ CK[i];

        // S盒应用
        tmp = (SM4_SBOX[(tmp >> 24) & 0xFF] << 24) |
            (SM4_SBOX[(tmp >> 16) & 0xFF] << 16) |
            (SM4_SBOX[(tmp >> 8) & 0xFF] << 8) |
            (SM4_SBOX[tmp & 0xFF]);

        // 线性变换L'
        tmp = tmp ^ rotl32(tmp, 13) ^ rotl32(tmp, 23);

        // 更新轮密钥
        k[0] = k[0] ^ tmp;
        rk[i] = k[0];

        // 循环移位
        uint32_t temp = k[0];
        k[0] = k[1];
        k[1] = k[2];
        k[2] = k[3];
        k[3] = temp;
    }
}

static uint32_t sm4_t(uint32_t x) {
    uint8_t b[4];
    b[0] = SM4_SBOX[(x >> 24) & 0xFF];
    b[1] = SM4_SBOX[(x >> 16) & 0xFF];
    b[2] = SM4_SBOX[(x >> 8) & 0xFF];
    b[3] = SM4_SBOX[x & 0xFF];

    uint32_t y = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];

    // 线性变换L
    return y ^ rotl32(y, 2) ^ rotl32(y, 10) ^ rotl32(y, 18) ^ rotl32(y, 24);
}

void sm4_encrypt_basic(const uint32_t rk[32], const uint8_t in[16], uint8_t out[16]) {
    uint32_t x[4];

    // 加载明文
    x[0] = ((uint32_t)in[0] << 24) | ((uint32_t)in[1] << 16) | ((uint32_t)in[2] << 8) | (uint32_t)in[3];
    x[1] = ((uint32_t)in[4] << 24) | ((uint32_t)in[5] << 16) | ((uint32_t)in[6] << 8) | (uint32_t)in[7];
    x[2] = ((uint32_t)in[8] << 24) | ((uint32_t)in[9] << 16) | ((uint32_t)in[10] << 8) | (uint32_t)in[11];
    x[3] = ((uint32_t)in[12] << 24) | ((uint32_t)in[13] << 16) | ((uint32_t)in[14] << 8) | (uint32_t)in[15];

    // 32轮加密
    for (int i = 0; i < 32; i++) {
        uint32_t tmp = x[1] ^ x[2] ^ x[3] ^ rk[i];
        tmp = sm4_t(tmp);
        x[0] = x[0] ^ tmp;

        // 循环移位
        if (i < 31) {
            uint32_t t = x[0];
            x[0] = x[1];
            x[1] = x[2];
            x[2] = x[3];
            x[3] = t;
        }
    }

    // 反序变换
    uint32_t tmp = x[0];
    x[0] = x[3];
    x[3] = tmp;
    tmp = x[1];
    x[1] = x[2];
    x[2] = tmp;

    // 存储密文
    out[0] = (x[0] >> 24) & 0xFF;
    out[1] = (x[0] >> 16) & 0xFF;
    out[2] = (x[0] >> 8) & 0xFF;
    out[3] = x[0] & 0xFF;
    out[4] = (x[1] >> 24) & 0xFF;
    out[5] = (x[1] >> 16) & 0xFF;
    out[6] = (x[1] >> 8) & 0xFF;
    out[7] = x[1] & 0xFF;
    out[8] = (x[2] >> 24) & 0xFF;
    out[9] = (x[2] >> 16) & 0xFF;
    out[10] = (x[2] >> 8) & 0xFF;
    out[11] = x[2] & 0xFF;
    out[12] = (x[3] >> 24) & 0xFF;
    out[13] = (x[3] >> 16) & 0xFF;
    out[14] = (x[3] >> 8) & 0xFF;
    out[15] = x[3] & 0xFF;
}

// ==================== T-Table优化实现 ====================
static uint32_t T0[256], T1[256], T2[256], T3[256];

void sm4_init_ttable() {
    for (int i = 0; i < 256; i++) {
        uint32_t s = SM4_SBOX[i];
        uint32_t t = s ^ rotl32(s, 2) ^ rotl32(s, 10) ^ rotl32(s, 18) ^ rotl32(s, 24);
        T0[i] = t;
        T1[i] = rotl32(t, 8);
        T2[i] = rotl32(t, 16);
        T3[i] = rotl32(t, 24);
    }
}

void sm4_encrypt_ttable(const uint32_t rk[32], const uint8_t in[16], uint8_t out[16]) {
    uint32_t x[4];

    // 加载明文
    x[0] = ((uint32_t)in[0] << 24) | ((uint32_t)in[1] << 16) | ((uint32_t)in[2] << 8) | (uint32_t)in[3];
    x[1] = ((uint32_t)in[4] << 24) | ((uint32_t)in[5] << 16) | ((uint32_t)in[6] << 8) | (uint32_t)in[7];
    x[2] = ((uint32_t)in[8] << 24) | ((uint32_t)in[9] << 16) | ((uint32_t)in[10] << 8) | (uint32_t)in[11];
    x[3] = ((uint32_t)in[12] << 24) | ((uint32_t)in[13] << 16) | ((uint32_t)in[14] << 8) | (uint32_t)in[15];

    // 32轮加密
    for (int i = 0; i < 32; i++) {
        uint32_t tmp = x[1] ^ x[2] ^ x[3] ^ rk[i];

        // 使用T-table
        tmp = T0[(tmp >> 24) & 0xFF] ^
            T1[(tmp >> 16) & 0xFF] ^
            T2[(tmp >> 8) & 0xFF] ^
            T3[tmp & 0xFF];

        x[0] ^= tmp;

        // 循环移位
        if (i < 31) {
            uint32_t t = x[0];
            x[0] = x[1];
            x[1] = x[2];
            x[2] = x[3];
            x[3] = t;
        }
    }

    // 反序变换和存储
    uint32_t tmp = x[0];
    x[0] = x[3];
    x[3] = tmp;
    tmp = x[1];
    x[1] = x[2];
    x[2] = tmp;

    // 存储密文
    for (int i = 0; i < 4; i++) {
        out[i * 4] = (x[i] >> 24) & 0xFF;
        out[i * 4 + 1] = (x[i] >> 16) & 0xFF;
        out[i * 4 + 2] = (x[i] >> 8) & 0xFF;
        out[i * 4 + 3] = x[i] & 0xFF;
    }
}

// ==================== AES-NI优化实现 ====================
#ifdef __AES__
// AES-NI S盒仿射变换常数
static const __m128i SM4_AFFINE = _mm_set_epi32(0xE19B0031, 0, 0, 0);

__m128i sm4_sbox_ni(__m128i x) {
    // 应用AES S盒
    x = _mm_aesenclast_si128(x, SM4_AFFINE);

    // 调整字节序
    return _mm_shuffle_epi8(x, _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
}

__m128i sm4_round_ni(__m128i block, __m128i rk) {
    __m128i t = _mm_xor_si128(block, rk);
    t = sm4_sbox_ni(t);

    // 线性变换L
    __m128i t2 = _mm_rotl_epi32(t, 2);
    __m128i t10 = _mm_rotl_epi32(t, 10);
    __m128i t18 = _mm_rotl_epi32(t, 18);
    __m128i t24 = _mm_rotl_epi32(t, 24);

    return _mm_xor_si128(_mm_xor_si128(t2, t10),
        _mm_xor_si128(t18, t24));
}

void sm4_encrypt_aesni(const uint32_t rk[32], const uint8_t in[16], uint8_t out[16]) {
    __m128i state = _mm_loadu_si128((const __m128i*)in);

    // 转换为大端序
    state = _mm_shuffle_epi8(state, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));

    // 32轮加密
    for (int i = 0; i < 32; i++) {
        __m128i round_key = _mm_set1_epi32(rk[i]);
        state = sm4_round_ni(state, round_key);

        // 移位操作
        if (i < 31) {
            state = _mm_alignr_epi8(state, state, 12);
        }
    }

    // 反序变换
    state = _mm_shuffle_epi32(state, _MM_SHUFFLE(0, 1, 2, 3));

    // 转换为小端序
    state = _mm_shuffle_epi8(state, _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));

    _mm_storeu_si128((__m128i*)out, state);
}
#endif

// ==================== AVX512+GFNI优化实现 ====================
#if defined(__AVX512F__) && defined(__GFNI__)
__m512i sm4_sbox_gfni(__m512i x) {
    // GFNI S盒仿射变换矩阵
    const __m512i affine_mat = _mm512_set1_epi64(0x1F1E1D1C1B1A1918);
    return _mm512_gf2p8affine_epi64_epi8(x, affine_mat, 0);
}

__m512i sm4_encrypt_block_gfni(__m512i block, const uint32_t rk[32]) {
    // 初始置换
    block = _mm512_shuffle_epi8(block, _mm512_set4_epi32(0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203));

    for (int i = 0; i < 32; i++) {
        __m512i round_key = _mm512_set1_epi32(rk[i]);
        __m512i t = _mm512_xor_si512(block, round_key);

        // GFNI S盒
        t = sm4_sbox_gfni(t);

        // 线性变换L
        __m512i t2 = _mm512_rol_epi32(t, 2);
        __m512i t10 = _mm512_rol_epi32(t, 10);
        __m512i t18 = _mm512_rol_epi32(t, 18);
        __m512i t24 = _mm512_rol_epi32(t, 24);

        __m512i l = _mm512_xor_si512(_mm512_xor_si512(t2, t10),
            _mm512_xor_si512(t18, t24));

        // 更新状态
        block = _mm512_alignr_epi32(block, l, 3);
    }

    // 最终置换
    block = _mm512_shuffle_epi32(block, _MM_PERM_DBCA);
    return _mm512_shuffle_epi8(block, _mm512_set4_epi32(0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C));
}

void sm4_encrypt_avx512_gfni(const uint32_t rk[32], const uint8_t* in, uint8_t* out, size_t blocks) {
    for (size_t i = 0; i < blocks; i += 4) {
        __m512i block = _mm512_loadu_si512((const __m512i*)(in + i * 16));
        block = sm4_encrypt_block_gfni(block, rk);
        _mm512_storeu_si512((__m512i*)(out + i * 16), block);
    }
}
#endif

// ==================== SM4-GCM实现 ====================
typedef struct {
    uint32_t rk[32];        // SM4轮密钥
    __m128i H;              // GHASH密钥
    __m128i J0;             // 初始计数器
    uint64_t aad_len;       // AAD长度（字节）
    uint64_t data_len;      // 数据长度（字节）
    __m128i tag;            // 认证标签
} sm4_gcm_ctx;

// 初始化GCM上下文
void sm4_gcm_init(sm4_gcm_ctx* ctx, const uint8_t* key, const uint8_t* iv, size_t iv_len) {
    // 生成SM4轮密钥
    sm4_key_schedule(key, ctx->rk);

    // 计算H = E_K(0)
    uint8_t zero[16] = { 0 };
    uint8_t H[16];
    sm4_encrypt_ttable(ctx->rk, zero, H);
    ctx->H = _mm_loadu_si128((const __m128i*)H);

    // 生成J0（计数器初始值）
    if (iv_len == 12) {
        // 标准IV处理
        ctx->J0 = _mm_set_epi32(0, 0, 0, 0x01000000);
        memcpy(&ctx->J0, iv, 12);
    }
    else {
        // GHASH处理长IV
        __m128i iv_block = _mm_setzero_si128();
        size_t blocks = iv_len / 16;
        size_t rem = iv_len % 16;

        for (size_t i = 0; i < blocks; i++) {
            __m128i data = _mm_loadu_si128((const __m128i*)(iv + i * 16));
            iv_block = _mm_xor_si128(iv_block, data);
            // GHASH乘法（简化实现）
            iv_block = _mm_clmulepi64_si128(iv_block, ctx->H, 0x00);
        }

        if (rem > 0) {
            uint8_t last[16] = { 0 };
            memcpy(last, iv + blocks * 16, rem);
            __m128i data = _mm_loadu_si128((const __m128i*)last);
            iv_block = _mm_xor_si128(iv_block, data);
        }

        // 添加长度信息
        __m128i len_block = _mm_set_epi64x(0, iv_len * 8);
        iv_block = _mm_xor_si128(iv_block, len_block);

        // 最终GHASH
        iv_block = _mm_clmulepi64_si128(iv_block, ctx->H, 0x00);
        ctx->J0 = iv_block;
    }

    ctx->aad_len = 0;
    ctx->data_len = 0;
    ctx->tag = _mm_setzero_si128();
}

// PCLMULQDQ加速的GHASH
__m128i ghash_pclmul(__m128i h, __m128i x) {
    __m128i z = _mm_clmulepi64_si128(x, h, 0x00);
    __m128i t = _mm_clmulepi64_si128(x, h, 0x11);
    __m128i r = _mm_xor_si128(z, t);

    // 模约简
    __m128i v = _mm_srli_si128(r, 8);
    r = _mm_xor_si128(r, _mm_slli_si128(v, 8));
    return _mm_xor_si128(r, _mm_srli_si128(v, 8));
}

// 更新GHASH状态
void ghash_update(sm4_gcm_ctx* ctx, const uint8_t* data, size_t len, int is_aad) {
    size_t blocks = len / 16;
    size_t rem = len % 16;

    for (size_t i = 0; i < blocks; i++) {
        __m128i block = _mm_loadu_si128((const __m128i*)(data + i * 16));
        ctx->tag = _mm_xor_si128(ctx->tag, block);
        ctx->tag = ghash_pclmul(ctx->H, ctx->tag);
    }

    if (rem > 0) {
        uint8_t last[16] = { 0 };
        memcpy(last, data + blocks * 16, rem);
        __m128i block = _mm_loadu_si128((const __m128i*)last);
        ctx->tag = _mm_xor_si128(ctx->tag, block);
        ctx->tag = ghash_pclmul(ctx->H, ctx->tag);
    }

    if (is_aad) {
        ctx->aad_len += len;
    }
    else {
        ctx->data_len += len;
    }
}

// CTR模式加密/解密
void ctr_encrypt(sm4_gcm_ctx* ctx, const uint8_t* in, uint8_t* out, size_t len) {
    uint32_t counter = _mm_cvtsi128_si32(ctx->J0);
    __m128i ctr_block = ctx->J0;

    for (size_t i = 0; i < len; i += 16) {
        // 加密计数器
        uint8_t enc_block[16];
        sm4_encrypt_ttable(ctx->rk, (uint8_t*)&ctr_block, enc_block);

        // 处理当前块
        size_t block_len = (len - i) < 16 ? (len - i) : 16;
        for (size_t j = 0; j < block_len; j++) {
            out[i + j] = in[i + j] ^ enc_block[j];
        }

        // 增加计数器
        counter++;
        ctr_block = _mm_insert_epi32(ctr_block, counter, 3);
    }
}

// 生成认证标签
void ghash_final(sm4_gcm_ctx* ctx, uint8_t tag[16]) {
    // 添加长度信息（AAD长度 + 数据长度）
    __m128i len_block = _mm_set_epi64x(ctx->aad_len * 8, ctx->data_len * 8);
    ctx->tag = _mm_xor_si128(ctx->tag, len_block);
    ctx->tag = ghash_pclmul(ctx->H, ctx->tag);

    // 加密J0生成标签掩码
    uint8_t tag_mask[16];
    sm4_encrypt_ttable(ctx->rk, (uint8_t*)&ctx->J0, tag_mask);

    // 计算最终标签
    __m128i mask = _mm_loadu_si128((const __m128i*)tag_mask);
    __m128i final_tag = _mm_xor_si128(ctx->tag, mask);
    _mm_storeu_si128((__m128i*)tag, final_tag);
}

// SM4-GCM加密
void sm4_gcm_encrypt(sm4_gcm_ctx* ctx,
    const uint8_t* key,          // 添加密钥参数
    const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* in, uint8_t* out, size_t len,
    uint8_t tag[16])
{
    // 初始化上下文 - 使用密钥和IV
    sm4_gcm_init(ctx, key, iv, iv_len);

    // 处理AAD
    if (aad_len > 0) {
        ghash_update(ctx, aad, aad_len, 1);
    }

    // 加密数据
    ctr_encrypt(ctx, in, out, len);

    // 更新GHASH（使用密文）
    ghash_update(ctx, out, len, 0);

    // 生成标签
    ghash_final(ctx, tag);
}

// ==================== 测试函数 ====================
void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02X", data[i]);
    }
    printf("\n");
}

int main() {
    // 测试密钥和明文
    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t plain[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t cipher[16];
    uint32_t rk[32];

    // 生成轮密钥
    sm4_key_schedule(key, rk);

    // 基础实现测试
    sm4_encrypt_basic(rk, plain, cipher);
    print_hex("Basic Cipher", cipher, 16);

    // T-table优化测试
    sm4_init_ttable();
    sm4_encrypt_ttable(rk, plain, cipher);
    print_hex("T-table Cipher", cipher, 16);

#ifdef __AES__
    // AES-NI优化测试
    sm4_encrypt_aesni(rk, plain, cipher);
    print_hex("AES-NI Cipher", cipher, 16);
#endif

#if defined(__AVX512F__) && defined(__GFNI__)
    // AVX512+GFNI优化测试
    uint8_t plain4[64];
    uint8_t cipher4[64];
    for (int i = 0; i < 4; i++) {
        memcpy(plain4 + i * 16, plain, 16);
    }
    sm4_encrypt_avx512_gfni(rk, plain4, cipher4, 4);
    print_hex("AVX512+GFNI Cipher", cipher4, 64);
#endif

    // SM4-GCM测试
    uint8_t iv[12] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B };
    uint8_t aad[20] = "AdditionalAuthData";
    uint8_t gcm_plain[64] = "This is a test message for SM4-GCM encryption!";
    uint8_t gcm_cipher[64];
    uint8_t gcm_tag[16];

    sm4_gcm_ctx gcm_ctx;
    sm4_gcm_encrypt(&gcm_ctx, key,   // 传入原始密钥
        iv, 12, aad, strlen((char*)aad),
        gcm_plain, gcm_cipher, strlen((char*)gcm_plain), gcm_tag);

    print_hex("GCM Cipher", gcm_cipher, strlen((char*)gcm_plain));
    print_hex("GCM Tag", gcm_tag, 16);

    return 0;
}