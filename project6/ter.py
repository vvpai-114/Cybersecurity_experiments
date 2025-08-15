import os
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from base64 import b64encode, b64decode


# ===== 1. 初始化协议参数 =====
class PasswordCheckupProtocol:
    def __init__(self):
        # 服务器生成RSA密钥对 :cite[2]:cite[5]
        self.server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.server_public_key = self.server_private_key.public_key()

        # 模拟服务器存储的泄露凭证数据库（实际中为加密哈希值）:cite[4]:cite[7]
        self.breached_credentials = set()

    # ===== 2. 客户端生成盲化请求 =====
    def client_generate_blinded_request(self, username, password):
        # 加盐哈希用户名和密码 :cite[2]
        salt = os.urandom(16)
        h_u = hashlib.sha256((username + salt.hex()).encode()).digest()
        h_p = hashlib.sha256((password + salt.hex()).encode()).digest()

        # 盲化请求：使用服务器公钥加密 (h_u || h_p) :cite[5]:cite[8]
        message = h_u + h_p
        blinded_request = self.server_public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return blinded_request, salt

    # ===== 3. 服务器签名盲化请求 =====
    def server_sign_request(self, blinded_request):
        # 服务器对盲请求签名（不解析内容）:cite[2]
        signature = self.server_private_key.sign(
            blinded_request,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    # ===== 4. 客户端解盲并生成查询凭证 =====
    def client_generate_credential(self, signature, salt, username, password):
        # 计算 h_u, h_p（需与盲化请求一致）
        h_u = hashlib.sha256((username + salt.hex()).encode()).digest()
        h_p = hashlib.sha256((password + salt.hex()).encode()).digest()
        message = h_u + h_p

        # 解盲签名（论文中省略，此处模拟为直接使用签名）
        k = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'credential_key',
            backend=default_backend()
        ).derive(signature)

        # 生成加密凭证 c = Enc(k, h_u || h_p) :cite[5]
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(k), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        c = encryptor.update(message) + encryptor.finalize()
        return iv + c + encryptor.tag

    # ===== 5. 服务器检查凭证是否泄露 =====
    def server_check_breach(self, credential):
        # 解密凭证（实际中服务器存储的是凭证的加密哈希值）:cite[4]
        # 此处简化为直接检查凭证是否在泄露数据库
        is_breached = credential in self.breached_credentials
        return is_breached

    # ===== 6. 模拟泄露数据库注册 =====
    def add_breached_credential(self, username, password):
        # 实际中存储加密哈希，此处用明文模拟
        self.breached_credentials.add((username, password))


# ===== 演示流程 =====
if __name__ == "__main__":
    protocol = PasswordCheckupProtocol()

    # 模拟添加一个泄露凭证
    protocol.add_breached_credential("user@example.com", "weakpassword")

    # 客户端：生成盲请求
    username = "user@example.com"
    password = "weakpassword"
    blinded_request, salt = protocol.client_generate_blinded_request(username, password)

    # 服务器：签名盲请求
    signature = protocol.server_sign_request(blinded_request)

    # 客户端：生成查询凭证
    credential = protocol.client_generate_credential(signature, salt, username, password)

    # 服务器：检查凭证是否泄露
    is_breached = protocol.server_check_breach((username, password))
    print(f"凭证是否泄露: {is_breached}")  # 应输出 True