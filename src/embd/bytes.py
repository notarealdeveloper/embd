"""
    pipe tensors between processes

    save tensors to persist stores
"""

__all__ = [
    'stdin_to_tensor',
    'bytes_to_tensor',
    'tensor_to_bytes',
    'tensor_to_stdout',
]

import sys
import numpy as np

# dtype for moving tensors between pipes, sockets, disk.
# dtype for computation is up to the user and the model.
IO_DTYPE = np.float32

def bytes_to_tensor(bytes):
    return np.frombuffer(bytes, dtype=IO_DTYPE)

def tensor_to_bytes(tensor):
    return tensor.astype(IO_DTYPE).tobytes()

def stdin_to_tensor():
    return bytes_to_tensor(sys.stdin.buffer.read())

def tensor_to_stdout(tensor):
    return sys.stdout.buffer.write(tensor_to_bytes(tensor))

