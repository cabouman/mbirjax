import numpy as np
import hashlib
import mbirjax as mj
import matplotlib.pyplot as plt

############################### CONTROL ###############################

file_path = "output/control_back_projection_recon_27c4a43a.npy"
expected_hash = "27c4a43acebc9e0e58a94ce67857b8b17a4aeb72b0146d494f6168e386dfb204"
control_back_projection = np.load(file_path)

hash_digest = hashlib.sha256(control_back_projection.tobytes()).hexdigest()
if hash_digest != expected_hash:
    print("\033[93m" + f"hash_digest of {file_path} is not the expected value." + "\033[0m")
    print("hash_digest:", hash_digest)

mj.slice_viewer(control_back_projection, slice_axis=2, title='Control Back Projection')


############################### SHARDED ###############################

file_path = "output/sharding_back_projection_recon_9470fe24.npy"
expected_hash = "9470fe2464cc4ce386fabb635c4914faf1e0b0a56d5ad465974bf35d4c4a2004"
sharded_back_projection = np.load(file_path)

hash_digest = hashlib.sha256(sharded_back_projection.tobytes()).hexdigest()
if hash_digest != expected_hash:
    print("\033[93m" + f"hash_digest of {file_path} is not the expected value." + "\033[0m")
    print("hash_digest:", hash_digest)

mj.slice_viewer(sharded_back_projection, slice_axis=2, title='Sharded Back Projection')


############################### DIFFERENCE ###############################

difference_back_projection = control_back_projection - sharded_back_projection
mj.slice_viewer(difference_back_projection, slice_axis=2, title='Difference Back Projection')

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