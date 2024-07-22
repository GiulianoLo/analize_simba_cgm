def print_ram_usage():
    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"RAM Usage: {ram_usage:.2f} MB")
