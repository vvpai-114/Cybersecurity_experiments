import os
import struct
import time
import mmap
import bisect
import math
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, cpu_count


# ========================
# SM3基础实现（修复版）
# ========================

class SM3Base:
    """
    SM3基础实现 - 符合国密标准GM/T 0004-2012
    """
    IV = [
        0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
        0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
    ]

    T = [0x79CC4519] * 16 + [0x7A879D8A] * 48

    @staticmethod
    def _left_rotate(x: int, n: int) -> int:
        """循环左移（修复负位移问题）"""
        n = n % 32  # 确保位移量在0-31之间
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

    @staticmethod
    def _ff(x: int, y: int, z: int, j: int) -> int:
        """布尔函数FF"""
        if j < 16:
            return x ^ y ^ z
        return (x & y) | (x & z) | (y & z)

    @staticmethod
    def _gg(x: int, y: int, z: int, j: int) -> int:
        """布尔函数GG"""
        if j < 16:
            return x ^ y ^ z
        return (x & y) | (~x & z)

    @staticmethod
    def _p0(x: int) -> int:
        """置换函数P0"""
        return x ^ SM3Base._left_rotate(x, 9) ^ SM3Base._left_rotate(x, 17)

    @staticmethod
    def _p1(x: int) -> int:
        """置换函数P1"""
        return x ^ SM3Base._left_rotate(x, 15) ^ SM3Base._left_rotate(x, 23)

    @staticmethod
    def _padding(data: bytes) -> bytes:
        """消息填充（修复版）"""
        length = len(data)
        bit_length = length * 8
        padding = b'\x80'  # 添加1字节的0x80

        # 计算需要填充的0的字节数
        pad_zeros = (56 - (length + 1) % 64) % 64
        padding += b'\x00' * pad_zeros

        # 添加64位的消息长度（大端序）
        padding += struct.pack('>Q', bit_length)
        return data + padding

    @staticmethod
    def _process_block(block: bytes, state: List[int]) -> List[int]:
        """处理一个512位消息块"""
        w = [0] * 68
        w_ = [0] * 64

        # 消息扩展
        for i in range(16):
            w[i] = struct.unpack('>I', block[i * 4:i * 4 + 4])[0]

        for j in range(16, 68):
            w[j] = SM3Base._p1(w[j - 16] ^ w[j - 9] ^ SM3Base._left_rotate(w[j - 3], 15)) ^ \
                   SM3Base._left_rotate(w[j - 13], 7) ^ w[j - 6]

        for j in range(64):
            w_[j] = w[j] ^ w[j + 4]

        # 压缩函数
        a, b, c, d, e, f, g, h = state

        for j in range(64):
            ss1 = SM3Base._left_rotate(
                (SM3Base._left_rotate(a, 12) + e + SM3Base._left_rotate(SM3Base.T[j], j % 32)) & 0xFFFFFFFF, 7)
            ss2 = ss1 ^ SM3Base._left_rotate(a, 12)
            tt1 = (SM3Base._ff(a, b, c, j) + d + ss2 + w_[j]) & 0xFFFFFFFF
            tt2 = (SM3Base._gg(e, f, g, j) + h + ss1 + w[j]) & 0xFFFFFFFF
            d = c
            c = SM3Base._left_rotate(b, 9)
            b = a
            a = tt1
            h = g
            g = SM3Base._left_rotate(f, 19)
            f = e
            e = SM3Base._p0(tt2)

        return [(x ^ y) & 0xFFFFFFFF for x, y in zip([a, b, c, d, e, f, g, h], state)]

    @classmethod
    def hash(cls, data: bytes) -> bytes:
        """计算SM3哈希值"""
        data = cls._padding(data)
        state = cls.IV.copy()

        # 处理消息块
        for i in range(0, len(data), 64):
            block = data[i:i + 64]
            state = cls._process_block(block, state)

        # 生成最终哈希值
        return b''.join(struct.pack('>I', x) for x in state)


# ========================
# SM3优化实现（修复版）
# ========================

class SM3Optimized(SM3Base):
    """
    SM3优化实现 - 包含多种优化策略
    """
    # 预计算T表（修复位移问题）
    T_CACHE = [SM3Base._left_rotate(T_val, j % 32) & 0xFFFFFFFF for j, T_val in enumerate(SM3Base.T)]

    # 预计算FF和GG函数的分段处理
    @staticmethod
    def _ff_optimized(x: int, y: int, z: int, j: int) -> int:
        """优化版布尔函数FF"""
        return x ^ y ^ z if j < 16 else (x & y) | (x & z) | (y & z)

    @staticmethod
    def _gg_optimized(x: int, y: int, z: int, j: int) -> int:
        """优化版布尔函数GG"""
        return x ^ y ^ z if j < 16 else (x & y) | (~x & z)

    @classmethod
    def _process_block_optimized(cls, block: bytes, state: List[int]) -> List[int]:
        """优化版消息块处理"""
        w = [0] * 68
        w_ = [0] * 64

        # 消息扩展优化：减少临时变量
        for i in range(16):
            w[i] = struct.unpack('>I', block[i * 4:i * 4 + 4])[0]

        for j in range(16, 68):
            # 优化P1计算
            temp = w[j - 16] ^ w[j - 9] ^ cls._left_rotate(w[j - 3], 15)
            w[j] = cls._p1(temp) ^ cls._left_rotate(w[j - 13], 7) ^ w[j - 6]

        for j in range(64):
            w_[j] = w[j] ^ w[j + 4]

        # 压缩函数优化：使用预计算的T表，减少位操作
        a, b, c, d, e, f, g, h = state

        for j in range(64):
            # 使用预计算的T表值
            t_val = cls.T_CACHE[j]

            # 优化SS1计算
            temp1 = cls._left_rotate(a, 12)
            temp2 = temp1 + e + t_val
            ss1 = cls._left_rotate(temp2, 7)
            ss2 = ss1 ^ temp1

            # 优化TT1和TT2计算
            tt1 = (cls._ff_optimized(a, b, c, j) + d + ss2 + w_[j]) & 0xFFFFFFFF
            tt2 = (cls._gg_optimized(e, f, g, j) + h + ss1 + w[j]) & 0xFFFFFFFF

            # 更新状态
            d = c
            c = cls._left_rotate(b, 9)
            b = a
            a = tt1
            h = g
            g = cls._left_rotate(f, 19)
            f = e
            e = cls._p0(tt2)

        return [(x ^ y) & 0xFFFFFFFF for x, y in zip([a, b, c, d, e, f, g, h], state)]

    @classmethod
    def hash(cls, data: bytes) -> bytes:
        """优化版哈希计算"""
        data = cls._padding(data)
        state = cls.IV.copy()

        # 处理消息块
        for i in range(0, len(data), 64):
            block = data[i:i + 64]
            state = cls._process_block_optimized(block, state)

        return b''.join(struct.pack('>I', x) for x in state)


# ========================
# 测试函数
# ========================

def test_sm3_implementation():
    """测试SM3实现"""
    test_cases = [
        (b"", "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b"),
        (b"abc", "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"),
        (b"abcd" * 16, "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"),
        (b"hello world", "44f0061e69fa6fdfc290c494654a05dc0c053da7e5c52b84ef93a9d67d3fff88")
    ]

    print("测试SM3实现:")
    print("=" * 80)
    print(f"{'输入':<20} {'预期输出':<66} {'实际输出':<66} {'状态'}")
    print("=" * 80)

    for data, expected in test_cases:
        # 测试基础实现
        result_base = SM3Base.hash(data).hex()
        status_base = "通过" if result_base == expected else "失败"
        print(f"{str(data):<20} {expected} {result_base} {status_base}")

        # 测试优化实现
        result_optimized = SM3Optimized.hash(data).hex()
        status_optimized = "通过" if result_optimized == expected else "失败"
        print(f"{str(data):<20} {expected} {result_optimized} {status_optimized}")

        print("-" * 80)


# ========================
# 性能测试函数
# ========================

def performance_test():
    """SM3实现性能测试"""
    print("\n性能测试:")
    print("=" * 80)
    print(f"{'数据大小':<10} {'实现方式':<15} {'耗时(ms)':<15} {'速度(MB/s)':<15}")
    print("=" * 80)

    # 测试不同大小的数据
    data_sizes = [10, 100, 1024, 10240, 102400, 1048576]  # 10B 到 1MB

    for size in data_sizes:
        data = os.urandom(size)

        # 测试基础实现
        start = time.time()
        SM3Base.hash(data)
        base_time = (time.time() - start) * 1000
        base_speed = (size / 1024 / 1024) / (base_time / 1000) if base_time > 0 else float('inf')
        print(f"{size:<10} {'基础':<15} {base_time:.4f} ms    {base_speed:.2f} MB/s")

        # 测试优化实现
        start = time.time()
        SM3Optimized.hash(data)
        optimized_time = (time.time() - start) * 1000
        optimized_speed = (size / 1024 / 1024) / (optimized_time / 1000) if optimized_time > 0 else float('inf')
        print(f"{size:<10} {'优化':<15} {optimized_time:.4f} ms    {optimized_speed:.2f} MB/s")

        if base_time > 0 and optimized_time > 0:
            speedup = base_time / optimized_time
            print(f"优化实现加速比: {speedup:.2f}x")

        print("-" * 80)


# ========================
# 主程序
# ========================

if __name__ == "__main__":
    # 测试SM3实现
    test_sm3_implementation()

    # 性能测试
    performance_test()

    # 大文件测试
    print("\n大文件测试:")
    data = b"a" * 5 * 1024 * 1024  # 5MB数据

    # 基础实现
    start = time.time()
    SM3Base.hash(data)
    base_time = (time.time() - start) * 1000
    print(f"基础实现处理5MB数据耗时: {base_time:.2f}ms")

    # 优化实现
    start = time.time()
    SM3Optimized.hash(data)
    optimized_time = (time.time() - start) * 1000
    print(f"优化实现处理5MB数据耗时: {optimized_time:.2f}ms")

    if base_time > 0 and optimized_time > 0:
        speedup = base_time / optimized_time
        print(f"优化实现加速比: {speedup:.2f}x")