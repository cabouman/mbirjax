import numpy as np
import hashlib
import mbirjax as mj
import matplotlib.pyplot as plt

############################### CONTROL ###############################

file_path = "output/control_back_projection_02d2da50.npy"
expected_hash = "02d2da5081fec0ef3705b4b53e8a484e53a30d8040e60396b0616778b3cf8125"
control_back_projection = np.load(file_path)

hash_digest = hashlib.sha256(control_back_projection.tobytes()).hexdigest()
if hash_digest != expected_hash:
    print("\033[93m" + f"hash_digest of {file_path} is not the expected value." + "\033[0m")
    print("hash_digest:", hash_digest)

mj.slice_viewer(control_back_projection, slice_axis=0, title='Control Back Projection')


############################### SHARDED ###############################

file_path = "output/sharded_back_projection_caa1f1ee.npy"
expected_hash = "caa1f1ee9324007a2256ee887289dc9ad2903bfb5aa2f2d49ecfe2fd0f97cf1e"
sharded_back_projection = np.load(file_path)

hash_digest = hashlib.sha256(sharded_back_projection.tobytes()).hexdigest()
if hash_digest != expected_hash:
    print("\033[93m" + f"hash_digest of {file_path} is not the expected value." + "\033[0m")
    print("hash_digest:", hash_digest)

mj.slice_viewer(sharded_back_projection, slice_axis=0, title='Sharded Back Projection')


############################### DIFFERENCE ###############################

difference_back_projection = control_back_projection - sharded_back_projection
mj.slice_viewer(difference_back_projection, slice_axis=0, title='Difference Back Projection')

# stats
print(f"Maximum Difference = {difference_back_projection.max()}")
print(f"Minimum Difference = {difference_back_projection.min()}")
print(f"Mean Difference = {difference_back_projection.mean()}")
print(f"Median Difference = {np.median(difference_back_projection)}")

# histogram
plt.hist(difference_back_projection.flatten(), bins=1000)
plt.title("Histogram of the values from the difference back projection")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()