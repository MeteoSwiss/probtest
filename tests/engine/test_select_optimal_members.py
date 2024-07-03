import random


def create_dummy_stats_file(filename, configurations, seed):
    random.seed(seed)
    # Create header
    max_time_dim = max([config["time_dim"] for config in configurations])
    time_header = ",".join([f"{t},{t},{t}" for t in range(max_time_dim)])
    header = [
        f"time,,,{time_header}",
        "statistic,," + ",max,mean,min" * max_time_dim,
        "file_ID,variable,height,,,,,,,,,",
    ]

    # Generate data rows
    data = []
    for config in configurations:
        time_dim = config["time_dim"]
        height_dim = config["height_dim"]
        variable = config["variable"]
        file_format = config["file_format"]

        for h in range(height_dim):
            row = f"{file_format},{variable},{h}.0"
            for t in range(time_dim):
                mean = round(random.uniform(0, 5), 2)
                max_val = mean + round(random.uniform(0, 1), 2)
                min_val = mean - round(random.uniform(0, 1), 2)
                row += f",{max_val},{mean},{min_val}"
            # Fill in remaining time slots with empty values if needed
            for _ in range(time_dim, max_time_dim):
                row += ",,,"
            data.append(row)

    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        for row in data:
            f.write(row + "\n")


if __name__ == "__main__":
    configurations = [
        {
            "time_dim": 3,
            "height_dim": 5,
            "variable": "v1",
            "file_format": "Format1:*test_3d*.nc",
        },
        {
            "time_dim": 3,
            "height_dim": 2,
            "variable": "v2",
            "file_format": "Format2:*test_2d*.nc",
        },
        {
            "time_dim": 2,
            "height_dim": 4,
            "variable": "v3",
            "file_format": "Format3:*test_2d*.nc",
        },
    ]
    # Create 50 files with different seeds
    seed = 42
    create_dummy_stats_file("stats_ref.csv", configurations, seed - 1)
    for i in range(1, 51):
        filename = f"stats_{i}.csv"
        create_dummy_stats_file(filename, configurations, seed + i)
