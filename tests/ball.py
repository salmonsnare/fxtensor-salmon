import numpy as np
from fxtensor_salmon import FXTensor

# -- Mapping of labels and indices --
# Define the state spaces
ball_domain = ['Black', 'White']
lamp_colors = ['Red', 'Green', 'Blue']
white_lamp_codomain = ['White lamp lights up', 'White lamp does not light up']

def main():
    # --- Representing Table 1 with FXTensor ---
    # The profile is [[input dimensions], [output dimensions]]
    # Input: ball color (2 types), Output: lamp color (3 types)
    profile_a = [[len(ball_domain)], [len(lamp_colors)]]
    # Data is represented as a Numpy array. The format is P(output|input)
    # data[input_index, output_index]
    data_a = np.array([
        # If the input is 'Black' (index 0)
        [0.1, 0.2, 0.7],  # Output: [P(Red|Black), P(Green|Black), P(Blue|Black)]
        # If the input is 'White' (index 1)
        [0.2, 0.3, 0.5]   # Output: [P(Red|White), P(Green|White), P(Blue|White)]
    ])
    tensor_a = FXTensor(profile_a, data=data_a)

    # --- Representing Table 2 with FXTensor ---
    # Input: lamp color (3 types), Output: white lamp state (2 types)
    profile_b = [[len(lamp_colors)], [len(white_lamp_codomain)]]
    data_b = np.array([
        # If the input is 'Red' (index 0)
        [0.8, 0.2],  # Output: [P(lights up|Red), P(does not light up|Red)]
        # If the input is 'Green' (index 1)
        [0.9, 0.1],  # Output: [P(lights up|Green), P(does not light up|Green)]
        # If the input is 'Blue' (index 2)
        [0.9, 0.1]   # Output: [P(lights up|Blue), P(does not light up|Blue)]
    ])
    tensor_b = FXTensor(profile_b, data=data_b)

    # --- Composition of tensors ---
    # The output of tensor_a is connected to the input of tensor_b
    result_tensor = tensor_a.composition(tensor_b)

    # --- Displaying the results ---
    print("Combined profile:", result_tensor.profile)
    print("Combined data (Numpy array):\n", result_tensor.data)
    print("\n--- Interpretation of results ---")
    for i, ball_color in enumerate(ball_domain):
        for j, lamp_status in enumerate(white_lamp_codomain):
            prob = result_tensor.data[i, j]
            print(f"P({lamp_status} | {ball_color}) = {prob:.2f}")

if __name__ == "__main__":
    main()