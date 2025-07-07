import sys

# Get all global variables
all_vars = list(globals().keys())

# Sort variables by size in descending order
all_vars_with_size = {var: sys.getsizeof(globals()[var]) for var in all_vars}
sorted_vars_with_size = sorted(all_vars_with_size.items(), key=lambda item: item[1], reverse=True)

# Print top variables by size
print("Top global variables by memory usage:")
for name, size in sorted_vars_with_size[:15]:  # Print top 15 or adjust as needed
    # Convert bytes to a human-readable format
    def format_bytes(size):
        power = 1024
        n = 0
        power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        while size > power and n <= len(power_labels):
            size /= power
            n += 1
        return f"{size:.2f} {power_labels[n]}"

    print(f"  {name}: {format_bytes(size)}")
