import numpy as np
from fxtensor_salmon import FXTensor

# Define labels for the state spaces
input_labels = ['A', 'B']
output_labels = ['X', 'Y']

# 1. Create the original tensor with string labels
profile = [[input_labels], [output_labels]]
original_data = np.array([[0.3, 0.7], [0.5, 0.5]])
original_tensor = FXTensor(profile, data=original_data)
print("Original Tensor:")
print(original_tensor)
print(original_tensor.data)
print("-" * 20)

# 2. Convert the tensor to JSON
json_data = original_tensor.to_json()
print("Generated JSON:")
print(json_data)
print("-" * 20)

# 3. Reconstruct a new tensor from the JSON data
reconstructed_tensor = FXTensor.from_json(json_data)
print("Reconstructed Tensor:")
print(reconstructed_tensor)
print(reconstructed_tensor.data)
print("-" * 20)

# 4. Verify that the original and reconstructed data are identical
np.testing.assert_array_equal(original_tensor.data, reconstructed_tensor.data)
assert original_tensor.labels == reconstructed_tensor.labels, "Labels do not match"

print("JSON I/O test passed: Original and reconstructed tensors (with labels) are identical.")