import h5py
import os

def copy_good_proteins(src_file, dest_file, good_proteins):
    # Remove the destination file if it already exists
    if os.path.isfile(dest_file):
        print(f"Removing existing {dest_file}...")
        os.remove(dest_file)

    with h5py.File(src_file, 'r') as src:
        with h5py.File(dest_file, 'a') as dest:
            for protein in good_proteins:
                group = dest.create_group(protein)
                
                # Copy attributes
                for attr in src[protein].attrs:
                    print(f"Copy attr: {attr}")
                    group.attrs[attr] = src[protein].attrs[attr]

                # Create the necessary hierarchy
                pybel_group = group.create_group('pybel')
                processed_group = pybel_group.create_group('processed')
                pdbbind_group = processed_group.create_group('pdbbind')

                data = src[f'{protein}/pybel/processed/pdbbind/data']
                print(data)
                # Copy the group/dataset from the source to destination file
                pdbbind_group.create_dataset('data', data=data)

                # Copy van_der_waals attribute if it exists
                if 'van_der_waals' in src[f'{protein}/pybel/processed/pdbbind'].attrs:
                    pdbbind_group.attrs['van_der_waals'] = src[f'{protein}/pybel/processed/pdbbind'].attrs['van_der_waals']




filename = "refined_2020_hdf_a2/pdbbind.hdf"
new_filename = "refined_2020_hdf_a2/good_proteins.hdf"
good_proteins = []

def check_protein(name, obj):
    global good_proteins
    if isinstance(obj, h5py.Dataset) and name.endswith('pdbbind/data'):
        # Add the name to the list of good proteins
        group_name = name.rsplit('/', 1)[0]
        good_proteins.append(group_name)  # Store the group name excluding 'pdbbind/data'

with h5py.File(filename, 'r') as f:
    f.visititems(check_protein)



print(f"Good proteins: {len(good_proteins)}")
print(f"First 5 good proteins: {good_proteins[:5]}")

# # Now copy the good proteins to a new HDF5 file
copy_good_proteins(filename, new_filename, good_proteins)
# print("="*50, "\n", "Good proteins")
# print_good_proteins(filename, good_proteins)