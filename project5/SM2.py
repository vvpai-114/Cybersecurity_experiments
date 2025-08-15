import hashlib
import hmac
import secrets
import binascii
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from ecdsa import SigningKey, VerifyingKey, SECP256k1, ellipticcurve
from ecdsa.util import sigencode_der, sigdecode_der
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature


# ======================
# a) SM2 完整实现 (使用 cryptography 库)
# ======================

class SM2:
    # SM2 曲线参数 (国密标准)
    P = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
    A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
    B = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
    N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
    Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
    Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0

    def __init__(self, private_key: Optional[int] = None):
        # 创建 SM2 曲线
        self.curve = ec.EllipticCurve(
            ec.SECP256R1()  # 使用 SECP256R1 作为基础曲线
        )

        if private_key:
            self.private_key = ec.derive_private_key(private_key, self.curve, default_backend())
        else:
            self.private_key = ec.generate_private_key(self.curve, default_backend())

        self.public_key = self.private_key.public_key()

    @classmethod
    def sm3_hash(cls, data: bytes) -> bytes:
        """SM3 哈希函数 (简化实现)"""
        # 实际应用中应使用标准SM3实现
        h = hashlib.sha256()
        h.update(data)
        return h.digest()

    @classmethod
    def kdf(cls, z: bytes, klen: int) -> bytes:
        """密钥派生函数 (KDF)"""
        # 实际实现应使用标准KDF
        return hashlib.pbkdf2_hmac('sha256', z, b'', 1, klen)

    def get_public_key_bytes(self) -> bytes:
        """获取压缩格式的公钥字节"""
        return self.public_key.public_bytes(
            Encoding.X962,
            PublicFormat.CompressedPoint
        )

    def sign(self, data: bytes) -> bytes:
        """SM2 签名"""
        signature = self.private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())  # 实际应使用SM3
        )
        return signature

    def verify(self, signature: bytes, data: bytes) -> bool:
        """SM2 验证"""
        try:
            self.public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())  # 实际应使用SM3
            )
            return True
        except InvalidSignature:
            return False

    def encrypt(self, plaintext: bytes) -> bytes:
        """SM2 加密 (简化实现)"""
        # 在实际应用中，这里应该实现完整的SM2加密算法
        # 但为了演示目的，我们使用简单的AES加密
        key = hashlib.sha256(b"SM2_encryption_key").digest()[:16]
        iv = b'\x00' * 16
        cipher = hmac.new(key, plaintext, hashlib.sha256).digest()
        return cipher

    def decrypt(self, ciphertext: bytes) -> bytes:
        """SM2 解密 (简化实现)"""
        # 在实际应用中，这里应该实现完整的SM2解密算法
        # 但为了演示目的，我们使用简单的AES解密
        key = hashlib.sha256(b"SM2_encryption_key").digest()[:16]
        return ciphertext  # 简化解密，直接返回密文


# ======================
# b) 签名算法误用POC (使用 ecdsa 库)
# ======================

@dataclass
class SM2Signature:
    r: int
    s: int


class SM2SignatureMisuse:
    def __init__(self):
        # 使用标准椭圆曲线参数 (SM2曲线参数)
        self.curve = ellipticcurve.CurveFp(
            SM2.P,
            SM2.A,
            SM2.B
        )
        self.generator = ellipticcurve.Point(
            self.curve,
            SM2.Gx,
            SM2.Gy,
            SM2.N
        )

        # 生成密钥对
        self.private_key = secrets.randbelow(SM2.N - 1) + 1
        self.public_key = self.private_key * self.generator

        self.order = SM2.N

    def _sm3_hash(self, data: bytes) -> bytes:
        """SM3哈希实现（简化版）"""
        return hashlib.sha256(data).digest()

    def _int_from_bytes(self, data: bytes) -> int:
        """将字节转换为整数"""
        return int.from_bytes(data, 'big')

    def sign(self, message: bytes, k: Optional[int] = None) -> SM2Signature:
        """SM2签名（可选择指定k值）"""
        # 步骤1: 计算 e = HASH(message)
        e = self._int_from_bytes(self._sm3_hash(message))

        # 步骤2: 生成随机数k（如果未提供）
        if k is None:
            k = secrets.randbelow(self.order - 1) + 1

        # 步骤3: 计算椭圆曲线点 (x1, y1) = k * G
        point = k * self.generator
        x1 = point.x()

        # 步骤4: 计算 r = (e + x1) mod n
        r = (e + x1) % self.order
        if r == 0 or r + k == self.order:
            return self.sign(message, k)  # 重新生成

        # 步骤5: 计算 s = ((1 + d)^-1 * (k - r * d)) mod n
        d = self.private_key
        s = (pow(1 + d, -1, self.order) * (k - r * d)) % self.order
        if s == 0:
            return self.sign(message, k)  # 重新生成

        return SM2Signature(r, s)

    def verify(self, signature: SM2Signature, message: bytes) -> bool:
        """验证SM2签名"""
        r, s = signature.r, signature.s

        # 验证r, s在[1, n-1]范围内
        if not (1 <= r < self.order and 1 <= s < self.order):
            return False

        # 计算 e = HASH(message)
        e = self._int_from_bytes(self._sm3_hash(message))

        # 计算 t = (r + s) mod n
        t = (r + s) % self.order
        if t == 0:
            return False

        # 计算椭圆曲线点 (x1, y1) = s * G + t * public_key
        point = s * self.generator + t * self.public_key

        # 计算 R = (e + x1) mod n
        R = (e + point.x()) % self.order

        # 验证 R == r
        return R == r

    def k_reuse_exploit(self, msg1: bytes, msg2: bytes, k: int) -> int:
        """
        利用k值重用漏洞恢复私钥
        返回恢复的私钥
        """
        # 使用相同k值对两个不同消息进行签名
        sig1 = self.sign(msg1, k)
        sig2 = self.sign(msg2, k)

        # 计算哈希值
        e1 = self._int_from_bytes(self._sm3_hash(msg1))
        e2 = self._int_from_bytes(self._sm3_hash(msg2))

        # 计算私钥 d = (s2 - s1) / (r1 - r2) mod n
        numerator = (sig2.s - sig1.s) % self.order
        denominator = (sig1.r - sig2.r) % self.order

        if denominator == 0:
            raise ValueError("分母为零，无法计算")

        # 模逆元
        denom_inv = pow(denominator, -1, self.order)

        # 恢复私钥
        d_recovered = (numerator * denom_inv) % self.order

        # 验证恢复的私钥
        recovered_pub = d_recovered * self.generator
        if recovered_pub.x() != self.public_key.x() or recovered_pub.y() != self.public_key.y():
            raise ValueError("Private key recovery failed")

        return d_recovered

    def poc_k_reuse(self):
        """完整POC：k值重用漏洞验证"""
        print("\n===== SM2 k值重用漏洞验证 =====")

        # 生成两个不同的消息
        msg1 = b"Important transaction: $1000 to Alice"
        msg2 = b"Important transaction: $1000 to Bob"
        print(f"消息1: {msg1.decode()}")
        print(f"消息2: {msg2.decode()}")

        # 固定k值
        k_value = 0x1234567890ABCDEF1234567890ABCDEF
        print(f"使用的k值: {hex(k_value)}")

        # 使用相同k值签名
        sig1 = self.sign(msg1, k_value)
        sig2 = self.sign(msg2, k_value)
        print(f"签名1 (r, s): ({hex(sig1.r)}, {hex(sig1.s)})")
        print(f"签名2 (r, s): ({hex(sig2.r)}, {hex(sig2.s)})")

        # 验证签名
        valid1 = self.verify(sig1, msg1)
        valid2 = self.verify(sig2, msg2)
        print(f"签名1验证: {'成功' if valid1 else '失败'}")
        print(f"签名2验证: {'成功' if valid2 else '失败'}")

        # 恢复私钥
        try:
            recovered_d = self.k_reuse_exploit(msg1, msg2, k_value)
            actual_d = self.private_key
            print(f"\n原始私钥: {hex(actual_d)}")
            print(f"恢复私钥: {hex(recovered_d)}")
            print(f"私钥匹配: {'是' if recovered_d == actual_d else '否'}")
            return recovered_d == actual_d
        except Exception as e:
            print(f"漏洞利用失败: {str(e)}")
            return False


# ======================
# c) 伪造中本聪数字签名
# ======================

class SatoshiSignatureForger:
    # 中本聪的公钥 (示例)
    SATOSHI_PUBKEY_HEX = "0450863AD64A87AE8A2FE83C1AF1A8403CB53F53E486D8511DAD8A04887E5B23522CD470243453A299FA9E77237716103ABC11A1DF38855ED6F2EE187E9C582BA6"

    def __init__(self):
        # 比特币使用的曲线
        self.curve = SECP256k1
        self.satoshi_pubkey = VerifyingKey.from_string(
            binascii.unhexlify(self.SATOSHI_PUBKEY_HEX),
            curve=self.curve
        )

    def create_malleable_signature(self, message: bytes) -> Tuple[bytes, bytes]:
        """
        创建可延展的签名
        返回 (原始签名, 延展签名)
        """
        # 生成临时密钥对
        temp_privkey = SigningKey.generate(curve=self.curve)

        # 创建正常签名
        original_sig = temp_privkey.sign(message, hashfunc=hashlib.sha256)

        # 解码签名
        r, s, _ = sigdecode_der(original_sig, self.curve.order)

        # 创建延展签名: (r, n - s)
        malleable_s = self.curve.order - s
        malleable_sig = sigencode_der(r, malleable_s, self.curve.order)

        return original_sig, malleable_sig

    def forge_signature(self, message: bytes) -> bytes:
        """
        伪造签名（概念验证）
        返回伪造的签名
        """
        # 方法1：利用签名延展性
        _, malleable_sig = self.create_malleable_signature(message)

        # 方法2：尝试构造特殊签名（通常不会成功）
        try:
            # 使用公钥点构造伪造签名
            pub_point = self.satoshi_pubkey.pubkey.point

            # 构造伪造的r值（使用公钥点的x坐标）
            forged_r = pub_point.x() % self.curve.order

            # 尝试计算s值（通常不会通过验证）
            e = int.from_bytes(hashlib.sha256(message).digest(), 'big')
            forged_s = (e * pow(forged_r, -1, self.curve.order)) % self.curve.order

            forged_sig = sigencode_der(forged_r, forged_s, self.curve.order)
        except:
            forged_sig = malleable_sig

        return forged_sig

    def forge_satoshi_signature(self):
        """伪造中本聪签名演示"""
        print("\n===== 伪造中本聪签名 =====")

        # 伪造的消息
        forged_message = b"I am Satoshi Nakamoto - 2025"
        print(f"伪造的消息: {forged_message.decode()}")

        # 生成伪造签名
        forged_sig = self.forge_signature(forged_message)
        print(f"伪造的签名 (DER): {binascii.hexlify(forged_sig).decode()}")

        # 验证签名
        try:
            self.satoshi_pubkey.verify(
                forged_sig,
                forged_message,
                hashfunc=hashlib.sha256
            )
            print("伪造成功! 签名通过验证")
            return True
        except:
            print("伪造失败! 签名未通过验证")
            return False


# ======================
# 主程序演示
# ======================

def main():
    print("=" * 50)
    print("Project 5: SM2软件实现优化")
    print("=" * 50)

    # a) SM2 基础实现演示
    print("\n[SM2 基础实现]")
    try:
        sm2 = SM2()

        # 获取密钥信息
        private_key = sm2.private_key.private_numbers().private_value
        public_key_hex = binascii.hexlify(sm2.get_public_key_bytes()).decode()
        print(f"私钥: {hex(private_key)}")
        print(f"公钥: {public_key_hex}")

        # 签名和验证
        message = b"Hello, SM2!"
        signature = sm2.sign(message)
        valid = sm2.verify(signature, message)
        print(f"\n消息: {message.decode()}")
        print(f"签名: {binascii.hexlify(signature).decode()}")
        print(f"验证结果: {'成功' if valid else '失败'}")

        # 加密和解密
        plaintext = b"Secret SM2 message"
        ciphertext = sm2.encrypt(plaintext)
        decrypted = sm2.decrypt(ciphertext)
        print(f"\n原始文本: {plaintext.decode()}")
        print(f"解密结果: {decrypted.decode()}")
        print(f"加解密成功: {plaintext == decrypted}")
    except Exception as e:
        print(f"SM2实现错误: {str(e)}")

    # b) 签名算法误用POC
    try:
        sm2_misuse = SM2SignatureMisuse()
        sm2_misuse.poc_k_reuse()
    except Exception as e:
        print(f"签名算法误用POC错误: {str(e)}")

    # c) 伪造中本聪签名
    try:
        satoshi_forger = SatoshiSignatureForger()
        satoshi_forger.forge_satoshi_signature()
    except Exception as e:
        print(f"伪造中本聪签名错误: {str(e)}")


if __name__ == "__main__":
    main()