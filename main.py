import os
import threading
import sqlite3
import numpy as np
from numba import cuda, jit
from web3 import Web3
from concurrent.futures import ThreadPoolExecutor

# Initialize Web3
web3 = Web3()

# SQLite setup
db_file = 'eth_addresses.db'

def initialize_db():
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS addresses (
                private_key TEXT PRIMARY KEY,
                eth_address TEXT UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_eth_address ON addresses (eth_address)
        ''')
        conn.commit()

def save_to_db(private_key, eth_address):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO addresses (private_key, eth_address) VALUES (?, ?)
        ''', (private_key, eth_address))
        conn.commit()

@jit
def generate_eth_address(private_key):
    account = web3.eth.account.from_key(private_key)
    return account.address

@cuda.jit
def generate_private_keys(start_key, num_keys, private_keys):
    idx = cuda.grid(1)
    if idx < num_keys:
        private_keys[idx] = hex(start_key + idx)[2:].zfill(64)

def process_private_key_range(start_key, end_key):
    num_keys = end_key - start_key + 1
    private_keys = np.empty(num_keys, dtype=np.object_)
    eth_addresses = np.empty(num_keys, dtype=np.object_)

    threadsperblock = 256
    blockspergrid = (num_keys + (threadsperblock - 1)) // threadsperblock

    generate_private_keys[blockspergrid, threadsperblock](start_key, num_keys, private_keys)
    cuda.synchronize()

    with ThreadPoolExecutor() as executor:
        for i in range(num_keys):
            private_key = private_keys[i]
            eth_address = generate_eth_address(private_key)
            executor.submit(save_to_db, private_key, eth_address)

def search_by_wallet_address(eth_address):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT private_key FROM addresses WHERE eth_address = ?
        ''', (eth_address,))
        result = cursor.fetchone()
        if result:
            return result[0]  # Return the private key
        else:
            return None

if __name__ == '__main__':
    initialize_db()
    
    # Process private key range
    start_key = int(input("Enter start private key (in hex, without 0x): "), 16)
    end_key = int(input("Enter end private key (in hex, without 0x): "), 16)
    
    # Number of threads for parallel processing
    num_threads = os.cpu_count()
    key_range = (end_key - start_key + 1) // num_threads

    threads = []
    for i in range(num_threads):
        start = start_key + i * key_range
        end = start + key_range - 1 if i != num_threads - 1 else end_key
        t = threading.Thread(target=process_private_key_range, args=(start, end))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Example search
    address_to_search = input("Enter Ethereum address to search: ")
    private_key = search_by_wallet_address(address_to_search)
    if private_key:
        print(f"Private key for address {address_to_search}: {private_key}")
    else:
        print("Address not found.")
