import numpy as np
import matplotlib.pyplot as plt

# Load data from NPZ file
data = np.load('data/dcrnn_predictions.npz')
predictions = data['predictions']
groundtruth = data['groundtruth']

# Flatten the 3D arrays to 1D
predictions_flat = predictions.flatten()
groundtruth_flat = groundtruth.flatten()

# Create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(groundtruth_flat, predictions_flat, alpha=0.5)
plt.plot([groundtruth_flat.min(), groundtruth_flat.max()], 
         [groundtruth_flat.min(), groundtruth_flat.max()], 
         'r--', lw=2)

plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Predictions vs Ground Truth')

# Calculate and display the correlation coefficient
correlation = np.corrcoef(groundtruth_flat, predictions_flat)[0, 1]
plt.annotate(f'Correlation: {correlation:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction')

plt.tight_layout()
plt.show()