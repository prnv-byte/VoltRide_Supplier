import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import onnx

os.makedirs("models", exist_ok=True)
os.makedirs("zkml_data_motor", exist_ok=True)

# 1. LOAD DATASET
print("Loading Motor Dataset...")
motor_df = pd.read_csv("datasets/VoltRide_Sequential_Industry_Motor_QC_10K_1.1.csv")

# 2. APPLY OEM CONSTRAINTS (Example constraints based on Motor metrics)
filtered_motor = motor_df[
    (motor_df["rated_power_W"] >= 1000) & (motor_df["rated_power_W"] <= 2000) &
    (motor_df["no_load_rpm"] >= 2500) & (motor_df["no_load_rpm"] <= 4000) &
    (motor_df["phase_winding_resistance_ohm"] <= 0.6) &
    (motor_df["efficiency_percent"] >= 75)
]

# 3. PREPARE DATA
features = [
    'rated_power_W', 'no_load_rpm', 'phase_winding_resistance_ohm', 
    'phase_imbalance_percent', 'torque_output_Nm', 'torque_ripple_percent', 
    'hall_sensor_voltage_V', 'hall_signal_integrity_percent', 'efficiency_percent'
]
X = filtered_motor[features].values
# Map target: PASS = 1.0, WARNING/FAIL = 0.0
y = np.where(filtered_motor["qc_status"] == "PASS", 1.0, 0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Save sample input for EZKL Proof generation
sample_input = X_tensor[0].reshape(1, -1).numpy().tolist()
with open("models/input_motor.json", "w") as f:
    json.dump({"input_data": sample_input}, f)

# 4. DEFINE MODEL
class MotorModel(nn.Module):
    def __init__(self):
        super(MotorModel, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = MotorModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. TRAIN MODEL
print("Training Motor Model...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/motor_verification_model.pth")

# 6. EXPORT TO ONNX (With EZKL Compatibility)
model.eval()
dummy_input = torch.randn(1, 9, dtype=torch.float32)

torch.onnx.export(
    model, (dummy_input,), "models/motor_verification_model.onnx", 
    export_params=True, input_names=["input"], output_names=["output"],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
    opset_version=14, do_constant_folding=True        
)

# Downgrade IR version for EZKL
onnx_model = onnx.load("models/motor_verification_model.onnx")
onnx_model.ir_version = 8  
onnx.save(onnx_model, "models/motor_verification_model.onnx")
print("✅ Motor model trained and ONNX exported.")