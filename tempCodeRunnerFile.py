import numpy as np

# Đọc dữ liệu từ file
data = np.load("data/dcrnn_predictions.npz")
predictions = data["predictions"]
groundtruth = data["groundtruth"]

# Tính MAE
import matplotlib.pyplot as plt

# Trung bình trên tất cả các nodes
avg_pred = np.mean(predictions, axis=2)  # (12, 6850)
avg_truth = np.mean(groundtruth, axis=2)  # (12, 6850)

# Vẽ biểu đồ tại thời điểm t=0 (bước đầu tiên)
plt.figure(figsize=(12, 6))
plt.plot(avg_truth[11], label="Ground Truth", linestyle="-", color="orange")
plt.plot(avg_pred[11], label="Prediction", linestyle="-", color="blue")

plt.xlabel("Sample (time)")
plt.ylabel("Speed")
plt.title("Prediction vs Ground Truth at t=0")
plt.legend()
plt.grid()
plt.show()
