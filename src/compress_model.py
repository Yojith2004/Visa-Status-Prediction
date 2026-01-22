import joblib
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, 'visa_processing_model_Random_Forest.pkl')
output_path = os.path.join(current_dir, 'visa_model_compressed.joblib')

print(f"Loading model from {input_path}...")
if not os.path.exists(input_path):
    print("Error: Input file not found!")
    exit(1)

model = joblib.load(input_path)
print("Model loaded.")

print(f"Compressing and saving to {output_path}...")
# Compress=3 provides a good balance of size reduction and speed
joblib.dump(model, output_path, compress=3)

original_size = os.path.getsize(input_path) / (1024 * 1024)
new_size = os.path.getsize(output_path) / (1024 * 1024)

print(f"Done! Original size: {original_size:.2f} MB")
print(f"Compressed size: {new_size:.2f} MB")
print(f"Reduction: {100 * (original_size - new_size) / original_size:.1f}%")
