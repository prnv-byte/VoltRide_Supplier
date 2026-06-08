import ezkl
import os
import asyncio

# -----------------------------
# File Paths
# -----------------------------
os.makedirs("zkml_data_chassis", exist_ok=True)

model_path = os.path.join("models", "chassis_verification_model.onnx")
data_path = os.path.join("models", "input_chassis.json")

settings_path = os.path.join("zkml_data_chassis", "settings.json")
compiled_model_path = os.path.join("zkml_data_chassis", "network.compiled")

pk_path = os.path.join("zkml_data_chassis", "test.pk")
vk_path = os.path.join("zkml_data_chassis", "test.vk")

proof_path = os.path.join("zkml_data_chassis", "test.pf")
witness_path = os.path.join("zkml_data_chassis", "witness.json")

srs_path = os.path.join("zkml_data_chassis", "kzg.srs")

async def execute_ezkl(func, *args, **kwargs):
    """
    Helper function to safely handle ezkl commands. 
    It forces Python to wait if the command is running as a background task.
    """
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result

async def run_zkml_pipeline():
    print("🚀 Starting ZKML Pipeline for Chassis Supplier...\n")

    # -----------------------------
    # Cleanup old/corrupt files
    # -----------------------------
    cleanup_files = [
        settings_path, compiled_model_path, pk_path, 
        vk_path, proof_path, witness_path, srs_path
    ]

    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)

    print("🧹 Old cryptographic files removed.\n")

    # -----------------------------
    # 1. Generate Settings
    # -----------------------------
    print("⚙️ Generating circuit settings...")
    await execute_ezkl(ezkl.gen_settings, model_path, settings_path, py_run_args=ezkl.PyRunArgs())
    print("✅ Settings generated.\n")

    # -----------------------------
    # 2. Calibrate
    # -----------------------------
    print("📏 Calibrating circuit margins...")
    await execute_ezkl(
        ezkl.calibrate_settings,
        data_path,
        model_path,
        settings_path,
        "resources",
        lookup_safety_margin=2,
        scales=[7],
        scale_rebase_multiplier=[1],
        max_logrows=12
    )
    print("✅ Calibration complete.\n")

    # -----------------------------
    # 3. Compile Circuit
    # -----------------------------
    print("🛠️ Compiling chassis verification circuit...")
    await execute_ezkl(ezkl.compile_circuit, model_path, compiled_model_path, settings_path)
    print("✅ Circuit compiled.\n")

   # -----------------------------
    # 4. Download SRS (BLOCKING FIX)
    # -----------------------------
    print("⬇️ Fetching Structured Reference String (SRS)...")
    await execute_ezkl(ezkl.get_srs, settings_path=settings_path, srs_path=srs_path, logrows=12)
    
    print("⏳ Waiting for Rust backend to finish downloading SRS...")
    
    # Wait until the file size is greater than 0 and stops changing
    prev_size = -1
    stable_count = 0
    while stable_count < 2:
        await asyncio.sleep(2)  # Pause for 2 seconds
        if os.path.exists(srs_path):
            curr_size = os.path.getsize(srs_path)
            if curr_size > 0 and curr_size == prev_size:
                stable_count += 1
            else:
                stable_count = 0
            prev_size = curr_size
        else:
            stable_count = 0
            
    print("✅ SRS fully downloaded and secured on disk.\n")

    # -----------------------------
    # 5. Generate Witness
    # -----------------------------
    print("🧾 Generating execution witness...")
    await execute_ezkl(
        ezkl.gen_witness,
        data_path, 
        compiled_model_path, 
        witness_path
    )
    print("✅ Witness generated.\n")

    # -----------------------------
    # 6. Setup Keys (PK/VK)
    # -----------------------------
    print("🔑 Generating Proving and Verification Keys...")
    await execute_ezkl(
        ezkl.setup,
        compiled_model_path, 
        vk_path, 
        pk_path, 
        witness_path=witness_path,
        srs_path=srs_path,
        disable_selector_compression=False
    )
    print("✅ PK and VK generated.\n")

    # -----------------------------
    # 7. Generate Proof
    # -----------------------------
    print("🧠 Generating Zero-Knowledge Execution Proof...")
    await execute_ezkl(
        ezkl.prove,
        witness_path, 
        compiled_model_path, 
        pk_path, 
        proof_path, 
        srs_path=srs_path
    )
    print("✅ Proof generated.\n")

    # -----------------------------
    # 8. Local Verification Check
    # -----------------------------
    print("🔍 Performing local verification check...")
    is_valid = await execute_ezkl(
        ezkl.verify,
        proof_path, 
        settings_path, 
        vk_path, 
        srs_path=srs_path, 
        reduced_srs=False
    )

    if is_valid:
        print("\n🎉 SUCCESS: Local Verification Passed 🎉")
        print("You can now send the following files to the OEM:")
        print(f" 1. {proof_path} (The Execution Proof)")
        print(f" 2. {vk_path} (The Verification Key)")
        print(f" 3. {settings_path} (Circuit Structure)")
        print(f" 4. {srs_path} (Reference String)\n")
    else:
        print("\n❌ Proof generation failed local verification.")

if __name__ == "__main__":
    asyncio.run(run_zkml_pipeline())