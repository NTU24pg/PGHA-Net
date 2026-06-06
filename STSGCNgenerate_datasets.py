import argparse
import os
import numpy as np


def generate_pghanet_io_data(data, x_offsets, y_offsets, input_dim=3):
    num_samples, num_nodes, num_features = data.shape

    if num_features < input_dim:
        raise ValueError(
            f"Expected at least {input_dim} input features, but got {num_features}."
        )

    x_data = data[:, :, :input_dim]
    y_data = data[:, :, 0:1]

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = num_samples - max(y_offsets)

    for t in range(min_t, max_t):
        x.append(x_data[t + x_offsets, ...])
        y.append(y_data[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


def generate_pghanet_train_val_test(args):
    data_seq = np.load(args.traffic_df_filename)["data"]

    x_offsets = np.arange(-(args.seq_length_x - 1), 1, 1)
    y_offsets = np.arange(args.y_start, args.seq_length_y + 1, 1)

    x, y = generate_pghanet_io_data(
        data=data_seq,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        input_dim=args.input_dim
    )

    print("x shape:", x.shape, "y shape:", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_x = locals()["x_" + split]
        split_y = locals()["y_" + split]

        print(split, "x:", split_x.shape, "y:", split_y.shape)

        np.savez_compressed(
            os.path.join(args.output_dir, f"{split}.npz"),
            x=split_x,
            y=split_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed/PEMS04/")
    parser.add_argument("--traffic_df_filename", type=str, default="data/PEMS04/PEMS04.npz")
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)

    args = parser.parse_args()

    generate_pghanet_train_val_test(args)
