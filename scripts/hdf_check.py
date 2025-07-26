import h5py

count = 0

def count_good_proteins(name, obj):
    global count
    if isinstance(obj, h5py.Group) and 'pdbbind/data' in name:
        print(f"Group with pdbbind/data found: {name}")
    # elif isinstance(obj, h5py.Dataset) and name.endswith('pdbbind/data') and 'affinity' in obj.attrs:
    #     count += 1
    #     # print(f"Dataset at pdbbind/data: {name}, {obj}")
    elif isinstance(obj, h5py.Group) and 'affinity' in obj.attrs and 'pybel/processed/pdbbind/data' in obj and 'van_der_waals' in obj['pybel/processed/pdbbind'].attrs:
        count += 1

filename = "pdbbind_2020_refined_good_proteins.hdf"


with h5py.File(filename, 'r') as f:
    print(f"Reading file: {filename}...")
    print(f"Proteins: {len(f.keys())}")

    f.visititems(count_good_proteins)

    print(f"Good proteins: {count}")