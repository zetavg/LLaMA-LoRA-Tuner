import os
import hashlib


def get_random_hex():
    random_bytes = os.urandom(16)
    hash_object = hashlib.sha256(random_bytes)
    hex_dig = hash_object.hexdigest()
    return hex_dig
