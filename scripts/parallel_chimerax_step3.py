import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Paths - update these!
input_root = r"FAST-master\data\PDBbind_v2020_refined\refined-set"
output_root = r"C:\Users\user\Documents\bindingaffinity\parallel_refined_2020_hydro_a2"
chimerax_exec = r"C:\Program Files\ChimeraX 1.10\bin\ChimeraX.exe"

def find_pdb_files(root_dir, proteins):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".pdb") and os.path.basename(dirpath) in proteins:
                yield os.path.join(dirpath, file)

def process_file(pdb_path):
    relative_path = os.path.relpath(pdb_path, input_root)
    relative_dir = os.path.dirname(relative_path)
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]

    output_dir = os.path.join(output_root, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, base_name + ".mol2")

    chimerax_cmd = f"""
    open "{pdb_path}";
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
        print(f"✅ Processed: {pdb_path} → {output_file}")
        if outs.strip():
            print(f"ChimeraX output:\n{outs}")
        if errs.strip():
            print(f"ChimeraX errors:\n{errs}")
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout expired for {pdb_path}, killing process.")
        proc.kill()
        outs, errs = proc.communicate()
        print(f"Process killed. Partial output:\n{outs}")
        if errs.strip():
            print(f"Errors:\n{errs}")

import os
import subprocess

# Paths - update these!
input_root = r"FAST-master\data\PDBbind_v2020_refined\refined-set"
refined_2020_pll_root = r"C:\Users\user\Documents\bindingaffinity\parallel_refined_2020_hydro"

def find_protein_complexes(root_dir):
    for dirs, _, filenames in os.walk(root_dir):
        yield os.path.basename(dirs)

if __name__ == "__main__":
    refined_proteins = list(find_protein_complexes(input_root))
    print(f"Len refined : {len(refined_proteins)}, {refined_proteins[:5]}, {refined_proteins[-5:]}")
    refined_2020_parallel = list(find_protein_complexes(refined_2020_pll_root))
    print(f"Len 2020 parallel refined : {len(refined_2020_parallel)}, {list(refined_2020_parallel)[:5]}, {list(refined_2020_parallel)[-5:]}")
    processed_proteins = set(refined_2020_parallel).intersection(set(refined_proteins))
    print(f"Processed proteins : {len(set(refined_2020_parallel).intersection(set(refined_proteins)) )}")
    remaining_proteins = set(refined_proteins).difference(set(refined_2020_parallel))
    print(f"Remaining for processing: {len(remaining_proteins)}")

    pdb_files = list(find_pdb_files(input_root, remaining_proteins))
    print(f"Found {len(pdb_files)} PDB files to process.")

    print(f"Save output to {output_root}")

    # Define the number of workers (parallel tasks)
    num_workers = 6  # Adjust this number based on your machine's capability

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map process_file function to all PDB files
        executor.map(process_file, pdb_files)

    print(f"DONE: Output saved to {output_root}")




