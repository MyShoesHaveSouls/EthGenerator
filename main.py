import numpy as np
import sqlite3
from numba import cuda
import ecdsa
import hashlib
import math
import memcache

@cuda.jit
def private_key_to_wallet_address_kernel(private_keys, wallet_addresses):
    idx = cuda.grid(1)
    if idx < private_keys.size:
        hex_key = private_keys[idx].decode()
        private_key_bytes = bytes.fromhex(hex_key)
        
        # Generate public key using ECDSA (secp256k1 curve)
        sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        public_key_bytes = vk.to_string()
        
        # Hash the public key using SHA-256
        sha256 = hashlib.sha256(public_key_bytes).digest()
        
        # Hash the result using RIPEMD-160
        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(sha256)
        wallet_address = ripemd160.hexdigest()
        
        wallet_addresses[idx] = wallet_address.encode()

def generate_wallet_addresses(private_keys):
    n = len(private_keys)

    # Allocate device arrays
    d_private_keys = cuda.to_device(np.array(private_keys, dtype=np.object))
    d_wallet_addresses = cuda.device_array(n, dtype=np.bytes_)

    # Optimal block and grid sizes
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)

    # Launch the kernel
    private_key_to_wallet_address_kernel[blocks_per_grid, threads_per_block](d_private_keys, d_wallet_addresses)

    # Copy the results back to the host
    wallet_addresses = d_wallet_addresses.copy_to_host()

    return [wa.decode() for wa in wallet_addresses]

def store_in_database(private_keys, wallet_addresses, memcache_client):
    conn = sqlite3.connect('wallet_addresses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wallets
                 (wallet_address TEXT PRIMARY KEY, private_key TEXT)''')
    
    for pk, wa in zip(private_keys, wallet_addresses):
        c.execute("INSERT OR REPLACE INTO wallets (wallet_address, private_key) VALUES (?, ?)",
                  (wa, pk))
        # Cache the wallet address and private key
        memcache_client.set(wa, pk)
        
    conn.commit()
    conn.close()

def get_private_key(wallet_address, memcache_client):
    private_key = memcache_client.get(wallet_address)
    if private_key:
        return private_key

    conn = sqlite3.connect('wallet_addresses.db')
    c = conn.cursor()
    c.execute("SELECT private_key FROM wallets WHERE wallet_address=?", (wallet_address,))
    result = c.fetchone()
    conn.close()

    if result:
        private_key = result[0]
        memcache_client.set(wallet_address, private_key)
        return private_key
    else:
        return None

def main(private_keys):
    memcache_client = memcache.Client(['127.0.0.1:11211'], debug=0)
    
    wallet_addresses = generate_wallet_addresses(private_keys)
    
    store_in_database(private_keys, wallet_addresses, memcache_client)
    
    print("Wallet addresses have been stored in the database and cache successfully.")

if __name__ == "__main__":
    private_keys = ['your_private_key_in_hex1', 'your_private_key_in_hex2', 'your_private_key_in_hexN']  # Replace with actual keys
    main(private_keys)
