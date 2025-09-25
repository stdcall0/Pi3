import os
import sys
import subprocess
import argparse
import re
from typing import List

def modify_pi3_file(pi3_path: str, original_content: str, mode: int) -> None:
    """
    Modifies the pi3.py file in memory and writes the new content.
    Uses regex to replace 'MODE = ...' line.
    """
    print(f"  -> Modifying '{os.path.basename(pi3_path)}' to set MODE = {mode}...")
    pattern = r"MODE\s*=\s*\d+"
    replacement = f"MODE = {mode}"
    new_content = re.sub(pattern, replacement, original_content)
    with open(pi3_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def run_example_script(cur_path: str, output_dir: str, data_path: str, mode: int) -> None:
    """
    Constructs and runs the example.py command, capturing its output.
    """
    example_script_path = os.path.join(cur_path, "example.py")
    
    save_path = os.path.join(output_dir, f"{mode}.glb")
    log_path = os.path.join(output_dir, f"{mode}.log")

    command = [
        sys.executable,
        example_script_path,
        "--data_path", data_path,
        "--save_path", save_path,
        "--conf", "20"
    ]
    
    print(f"  -> Executing command: {' '.join(command)}")
    print(f"  -> Saving stdout to '{os.path.relpath(log_path, cur_path)}'...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("--- STDOUT ---\n")
        log_file.write(result.stdout)
        log_file.write("\n\n--- STDERR ---\n")
        log_file.write(result.stderr)
    
    if result.returncode == 0:
        print(f"  -> Successfully generated '{os.path.relpath(save_path, cur_path)}'.")
    else:
        print(f"  -> WARNING: Command for mode {mode} finished with exit code {result.returncode}.")
        print(f"     Check '{os.path.relpath(log_path, cur_path)}' for details.")


def main():
    # 1. Setup argparse
    parser = argparse.ArgumentParser(
        description="Run pi3.py example script for multiple modes."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing input images (consistent with example.py)."
    )
    args = parser.parse_args()
    
    data_path = args.data_path

    # 2. Define paths
    modes: List[int] = [0, 1, 2, 3, 4]
    cur_path = os.path.dirname(os.path.abspath(__file__))
    pi3_path = os.path.join(cur_path, "pi3", "models", "pi3.py")
    example_py_path = os.path.join(cur_path, "example.py")

    output_dir = os.path.join(cur_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Check if required files/dirs exist
    if not os.path.isfile(pi3_path):
        print(f"Error: The target file '{pi3_path}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(example_py_path):
        print(f"Error: The script to run '{example_py_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(data_path):
        print(f"Error: The specified data path '{data_path}' does not exist.")
        sys.exit(1)
        
    print(f"Found pi3.py at: {pi3_path}")
    print(f"Outputs will be saved to: {output_dir}")
    print("-" * 40)

    # 4. Read the original content of pi3.py
    with open(pi3_path, 'r', encoding='utf-8') as f:
        original_pi3_content = f.read()

    try:
        # 5. Loop through each mode
        for mode in modes:
            print(f"Processing MODE = {mode}...")
            modify_pi3_file(pi3_path, original_pi3_content, mode)
            
            run_example_script(cur_path, output_dir, data_path, mode)
            
            print("-" * 40)

    finally:
        # 6. Restore the original content of pi3.py
        print("Restoring original pi3.py file...")
        with open(pi3_path, 'w', encoding='utf-8') as f:
            f.write(original_pi3_content)
        print("Done.")


if __name__ == "__main__":
    main()