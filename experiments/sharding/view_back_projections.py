import numpy as np
import hashlib
import mbirjax as mj

file_path = "output/control_back_projection_02d2da50.npy"
expected_hash = "02d2da5081fec0ef3705b4b53e8a484e53a30d8040e60396b0616778b3cf8125"
back_projection_normal = np.load(file_path)

hash_digest = hashlib.sha256(back_projection_normal.tobytes()).hexdigest()
if hash_digest != expected_hash:
    print("\033[93m" + f"hash_digest of {file_path} is not the expected value." + "\033[0m")
    print("hash_digest:", hash_digest)

mj.slice_viewer(back_projection_normal, slice_axis=0, title='Control Back Projection')