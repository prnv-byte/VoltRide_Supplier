import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import onnx
import sys

# 1. LOAD DATASET
print("Loading Battery Dataset...")
battery_df = pd.read_csv("datasets/VoltRide_Sequential_Battery_QC_10K_1.1.csv")
TOTAL_STOCK = len(battery_df)

# 2. APPLY OEM CONSTRAINTS
filtered_battery = battery_df[
    (battery_df["nominal_voltage_V"] >= 2.5) & (battery_df["nominal_voltage_V"] <= 4.5) &
    (battery_df["internal_resistance_mOhm"] >= 0) & (battery_df["internal_resistance_mOhm"] <= 500) &
    (battery_df["capacity_mAh"] >= 1000) & (battery_df["capacity_mAh"] <= 500000) &
    (battery_df["SOH_percent"] >= 0) & (battery_df["SOH_percent"] <= 100) &
    (battery_df["self_discharge_rate_percent_per_month"] >= 0) & (battery_df["self_discharge_rate_percent_per_month"] <= 10) &
    (battery_df["temperature_delivery_C"] >= -40) & (battery_df["temperature_delivery_C"] <= 85)
]

# Dynamic Stock Check
ORDER_SIZE = 400
matched_count = len(filtered_battery)

# --- NEW GRACEFUL EXIT LOGIC ---
if matched_count == 0:
    print("\n❌ 0 batteries matching the requested constraints are available in stock.")
    print("Cannot proceed with ML training or manifest generation. Process aborted.\n")
    sys.exit(0)  # Cleanly stops the script here without throwing a Python error
# -------------------------------

elif matched_count < ORDER_SIZE:
    print(f"Only {matched_count} number of batteries were able to be fetched from the total stock.")
    print(f"The model will be trained for only {matched_count} number of batteries.")
    ORDER_SIZE = matched_count
else:
    print(f"Successfully fetched {ORDER_SIZE} batteries matching OEM specifications from the total stock.")

# Select the target order size and save the physical manifest for the OEM
final_batch = filtered_battery.iloc[:ORDER_SIZE].copy()
os.makedirs("models", exist_ok=True)
final_batch.to_csv("models/oem_battery_delivery_batch.csv", index=False)
print(f"Generated 'oem_battery_delivery_batch.csv' for shipping manifest.")

# 3. EXTRACT FEATURES & GENERATE TARGETS
features = final_batch[[
    "nominal_voltage_V", "internal_resistance_mOhm", "capacity_mAh", 
    "SOH_percent", "temperature_delivery_C", "self_discharge_rate_percent_per_month"
]]

def generate_label(row):
    if row["SOH_percent"] > 80 and row["temperature_delivery_C"] < 50: 
        return 1
    else:
        return 0

final_batch["Target"] = final_batch.apply(generate_label, axis=1)

X = features.values
y = final_batch["Target"].values

# 4. PREPROCESS & SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 5. BATTERY MODEL ARCHITECTURE (6 Inputs)
class BatteryVerificationMLP(nn.Module):
    def __init__(self):
        super(BatteryVerificationMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = BatteryVerificationMLP()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Training Battery Model on {ORDER_SIZE} Validated Units...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/battery_verification_model.pth")
print("Battery model trained and saved.")

# 6. EXPORT TO ONNX
model.eval() 
dummy_input = torch.randn(1, 6, dtype=torch.float32)

torch.onnx.export(
    model,                          
    (dummy_input,),
    "models/battery_verification_model.onnx", 
    export_params=True,             
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
    opset_version=14,               
    do_constant_folding=True        
)

# Downgrade the IR Version for EZKL Compatibility
onnx_model = onnx.load("models/battery_verification_model.onnx")
onnx_model.ir_version = 8  
onnx.save(onnx_model, "models/battery_verification_model.onnx")
print("ONNX model exported and IR version downgraded for EZKL!")

# 7. GENERATE ZKML JSON DATA
sample_input = X_tensor[0:1] 
sample_output = model(sample_input).detach()

data_json = {
    "input_data": sample_input.numpy().tolist(),
    "output_data": sample_output.numpy().tolist()
}

with open("models/input.json", "w") as f:
    json.dump(data_json, f)
print("ZKML Calibration data saved to models/input.json")