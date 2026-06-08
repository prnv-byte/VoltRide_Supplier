import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import sys
import onnx

# 1. LOAD DATASET
print("Loading Chassis Dataset...")
# Assumes the dataset is in the 'datasets' folder to match the battery script structure
chassis_df = pd.read_csv("datasets/VoltRide_Sequential_Chassis_10k_1.1.csv")
TOTAL_STOCK = len(chassis_df)

# 2. APPLY OEM CONSTRAINTS (Chassis-specific)
filtered_chassis = chassis_df[
    (chassis_df["Weld_Quality_Score"] >= 75) &
    (chassis_df["Frame_Weight_kg"] >= 80) & (chassis_df["Frame_Weight_kg"] <= 90) &
    (chassis_df["Dimensional_Deviation_mm"] <= 0.6) &
    (chassis_df["Static_Load_Capacity_kg"] >= 400)
]

# Dynamic Stock Check
ORDER_SIZE = 400
matched_count = len(filtered_chassis)

# --- NEW GRACEFUL EXIT LOGIC ---
if matched_count == 0:
    print("\n❌ 0 chassis matching the requested constraints are available in stock.")
    print("Cannot proceed with ML training or manifest generation. Process aborted.\n")
    sys.exit(0)  # Cleanly stops the script here without throwing a Python error
# -------------------------------

elif matched_count < ORDER_SIZE:
    print(f"Only {matched_count} chassis were able to be fetched from the total stock.")
    print(f"The model will be trained for only {matched_count} chassis.")
    ORDER_SIZE = matched_count
else:
    print(f"Successfully fetched {ORDER_SIZE} chassis matching OEM specifications from the total stock.")

# Select the target order size and save the physical manifest for the OEM
final_batch = filtered_chassis.iloc[:ORDER_SIZE].copy()
os.makedirs("models", exist_ok=True)
final_batch.to_csv("models/oem_chassis_delivery_batch.csv", index=False)
print("Generated 'oem_chassis_delivery_batch.csv' for shipping manifest.")

# 3. EXTRACT FEATURES & GENERATE TARGETS
features = final_batch[[
    "Weld_Quality_Score", "Frame_Weight_kg", "Dimensional_Deviation_mm", 
    "Hardness_BHN", "Surface_Defect_Count", "Static_Load_Capacity_kg"
]]

def generate_label(row):
    # Custom ML target logic: Passed standard QC and has minimal surface defects
    if row["Overall_QC_Status"] == "Pass" and row["Surface_Defect_Count"] <= 5: 
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

# 5. CHASSIS MODEL ARCHITECTURE (6 Inputs to match the battery model structure)
class ChassisVerificationMLP(nn.Module):
    def __init__(self):
        super(ChassisVerificationMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = ChassisVerificationMLP()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Training Chassis Model on {ORDER_SIZE} Validated Units...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/chassis_verification_model.pth")
print("Chassis model trained and saved.")

# 6. EXPORT TO ONNX
model.eval() 
dummy_input = torch.randn(1, 6, dtype=torch.float32)

torch.onnx.export(
    model,                          
    (dummy_input,),
    "models/chassis_verification_model.onnx", 
    export_params=True,             
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
    opset_version=14,               
    do_constant_folding=True        
)

# Downgrade the IR Version for EZKL Compatibility
onnx_model = onnx.load("models/chassis_verification_model.onnx")
onnx_model.ir_version = 8  
onnx.save(onnx_model, "models/chassis_verification_model.onnx")
print("ONNX model exported and IR version downgraded for EZKL!")

# 7. GENERATE ZKML JSON DATA
sample_input = X_tensor[0:1] 
sample_output = model(sample_input).detach()

data_json = {
    "input_data": sample_input.numpy().tolist(),
    "output_data": sample_output.numpy().tolist()
}

with open("models/input_chassis.json", "w") as f:
    json.dump(data_json, f)
print("ZKML Calibration data saved to models/input_chassis.json")