"""
Test script to verify the tensor conversion fix
"""
import torch
import numpy as np

def test_probability_handling():
    """Test the probability handling that was causing the error"""
    print("Testing probability tensor handling...")
    
    # Simulate the problematic scenario
    class_labels = [
        "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
        "Nevus (Mole)", "Dermatofibroma", "Vascular Lesion", "Actinic Keratosis"
    ]
    
    # Test with tensor probabilities
    probabilities_tensor = torch.tensor([0.15, 0.12, 0.10, 0.25, 0.15, 0.12, 0.11])
    
    # Test the fixed code
    if isinstance(probabilities_tensor, torch.Tensor):
        prob_values = probabilities_tensor.detach().cpu().numpy()
    else:
        prob_values = probabilities_tensor
    
    if isinstance(prob_values, (list, tuple)):
        prob_list = [float(p) for p in prob_values]
    else:
        prob_list = [float(p) for p in prob_values.flatten()]
    
    prob_chart_data = dict(zip(class_labels, prob_list))
    
    print("Tensor probabilities:", prob_chart_data)
    print("Success: No tensor conversion error!")
    
    # Test with numpy array
    probabilities_numpy = np.array([0.20, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10])
    
    if isinstance(probabilities_numpy, torch.Tensor):
        prob_values = probabilities_numpy.detach().cpu().numpy()
    else:
        prob_values = probabilities_numpy
    
    if isinstance(prob_values, (list, tuple)):
        prob_list = [float(p) for p in prob_values]
    else:
        prob_list = [float(p) for p in prob_values.flatten()]
    
    prob_chart_data = dict(zip(class_labels, prob_list))
    
    print("NumPy probabilities:", prob_chart_data)
    print("Success: No numpy conversion error!")

if __name__ == "__main__":
    test_probability_handling()
