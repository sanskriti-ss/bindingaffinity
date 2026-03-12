import os
import subprocess

# Paths - update these!
input_roots = [
    r"C:\bindingaffinity\refined-set",
]
output_root = r"C:\bindingaffinity\new_processed_2020"
chimerax_exec = r"C:\Program Files\ChimeraX 1.9\bin\ChimeraX.exe"

def find_structure_files(root_dir):
    """Find ligand (.sdf) and pocket (.pdb) files, excluding protein files."""
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            file_lower = file.lower()
            # Include .sdf files with 'ligand' in name, or .pdb files with 'pocket' in name
            if (file_lower.endswith(".sdf") and "ligand" in file_lower) or \
               (file_lower.endswith(".pdb") and "pocket" in file_lower):
                yield os.path.join(dirpath, file)

# def process_file(pdb_path):
#     relative_path = os.path.relpath(pdb_path, input_root)
#     relative_dir = os.path.dirname(relative_path)
#     base_name = os.path.splitext(os.path.basename(pdb_path))[0]

#     output_dir = os.path.join(output_root, relative_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     output_file = os.path.join(output_dir, base_name + ".mol2")

#     chimerax_cmd = f"""
#     open "{pdb_path}";
#     delete H;
#     addh useGluName true;
#     addcharge;
#     save "{output_file}" format mol2;
#     close session;
#     """

#     try:
#         completed_process = subprocess.run(
#             [chimerax_exec, "--nogui", "--cmd", chimerax_cmd],
#             check=True,
#             capture_output=True,
#             text=True,
#             timeout=300  # 5 minutes timeout
#         )
#         print(f"✅ Processed: {pdb_path} → {output_file}")
#         if completed_process.stdout.strip():
#             print(f"ChimeraX output:\n{completed_process.stdout}")
#         if completed_process.stderr.strip():
#             print(f"ChimeraX errors:\n{completed_process.stderr}")
#     except subprocess.TimeoutExpired:
#         print(f"⏰ Timeout expired for {pdb_path}. Skipping to next file.")
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Error processing {pdb_path}:\n{e.stderr}")

# def main():
#     pdb_files = list(find_pdb_files(input_root))
#     print(f"Found {len(pdb_files)} PDB files to process.")

#     for i, pdb in enumerate(pdb_files, 1):
#         print(f"\nProcessing file {i} of {len(pdb_files)}:")
#         process_file(pdb)

# if __name__ == "__main__":
#     main()


import subprocess
import threading

def process_file(structure_path, input_root):
    relative_path = os.path.relpath(structure_path, input_root)
    relative_dir = os.path.dirname(relative_path)
    base_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Preserve the year range folder structure in output
    year_range = os.path.basename(input_root)
    output_dir = os.path.join(output_root, year_range, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, base_name + ".mol2")

    chimerax_cmd = f"""
    open "{structure_path}";
    delete H;
    addh useGluName true;
    addcharge;
    save "{output_file}" format mol2;
    close session;
    exit
    """

    proc = subprocess.Popen(
        [chimerax_exec, "--nogui", "--cmd", chimerax_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        outs, errs = proc.communicate(timeout=300)  # wait 5 minutes max
        print(f"✅ Processed: {structure_path} → {output_file}")
        if outs.strip():
            print(f"ChimeraX output:\n{outs}")
        if errs.strip():
            print(f"ChimeraX errors:\n{errs}")
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout expired for {structure_path}, killing process.")
        proc.kill()
        outs, errs = proc.communicate()
        print(f"Process killed. Partial output:\n{outs}")
        if errs.strip():
            print(f"Errors:\n{errs}")

def main():
    for input_root in input_roots:
        print(f"\n{'='*60}")
        print(f"Processing directory: {input_root}")
        print(f"{'='*60}\n")
        
        if not os.path.exists(input_root):
            print(f"⚠️ Warning: Directory not found: {input_root}")
            continue
        
        structure_files = list(find_structure_files(input_root))
        print(f"Found {len(structure_files)} ligand (.sdf) and pocket (.pdb) files to process in {os.path.basename(input_root)}.")
        
        for i, structure_file in enumerate(structure_files, 1):
            print(f"\nProcessing file {i} of {len(structure_files)}:")
            process_file(structure_file, input_root)
    
    print(f"\n{'='*60}")
    print("All directories processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
