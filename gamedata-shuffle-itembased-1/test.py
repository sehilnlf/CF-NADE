import h5py

# Đường dẫn tới file HDF5
file_path = "game-ratings.hdf5"

# Mở file
with h5py.File(file_path, 'r') as f:
    # In tất cả các nhóm và datasets
    def print_structure(name, obj):
        print(name, obj)

    f.visititems(print_structure)

    # Kiểm tra dataset chứa `true_r` hoặc `out_r`
    dataset_name = "name_of_dataset"  # Thay thế bằng tên dataset
    data = f[dataset_name][:]
    print(f"Data shape: {data.shape}")
    print(f"Data sample:\n{data[:10]}")
