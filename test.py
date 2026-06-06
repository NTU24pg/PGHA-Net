import argparse
import torch
import util
import numpy as np
import pywt
from model import PGHANet


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0001,
    help="weight decay rate"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoints/PGHANet/PEMS08/best_model.pth",
    help=""
)
args = parser.parse_args()


def flow_decoupling_layer(x, wavelet_basis, decomposition_level):
    x = x.transpose(0, 3, 2, 1)
    coefficients = pywt.wavedec(x, wavelet_basis, level=decomposition_level)

    low_frequency_coefficients = [coefficients[0]]
    for _ in range(len(coefficients) - 1):
        low_frequency_coefficients.append(None)

    high_frequency_coefficients = [None]
    for i in range(len(coefficients) - 1):
        high_frequency_coefficients.append(coefficients[i + 1])

    trend_signal = pywt.waverec(low_frequency_coefficients, wavelet_basis).transpose(0, 3, 2, 1)
    event_signal = pywt.waverec(high_frequency_coefficients, wavelet_basis).transpose(0, 3, 2, 1)

    return trend_signal, event_signal


wavelet_basis = "coif1"
decomposition_level = 1


def main():
    if args.data == "PEMS08":
        args.data = "data//" + args.data
        num_nodes = 170
        granularity = 288
        channels = 96

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        num_nodes = 358
        granularity = 288
        channels = 32

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        num_nodes = 307
        granularity = 288
        channels = 64

    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        num_nodes = 883
        granularity = 288
        channels = 128

    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32

    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32

    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    else:
        raise ValueError("Unknown dataset: {}".format(args.data))

    device = torch.device(args.device)

    model = PGHANet(
        device,
        args.input_dim,
        num_nodes,
        channels,
        granularity,
        args.dropout
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("PGHA-Net model loaded successfully")

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size
    )

    scaler = dataloader["scaler"]
    predictions = []

    ground_truth = torch.Tensor(dataloader["y_test"]).to(device)
    ground_truth = ground_truth.transpose(1, 3)[:, 0, :, :]

    for _, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        historical_flow = x[:, :, :, 0]
        historical_flow = historical_flow[:, :, :, np.newaxis]
        historical_flow = scaler.inverse_transform(historical_flow)

        future_flow = y[:, :, :, 0]
        future_flow = future_flow[:, :, :, np.newaxis]

        temporal_features = x[:, :, :, 1:3]

        trend_signal, event_signal = flow_decoupling_layer(
            historical_flow,
            wavelet_basis,
            decomposition_level
        )

        trend_signal = scaler.transform(trend_signal)
        event_signal = scaler.transform(event_signal)

        trend_input = torch.Tensor(trend_signal).to(device)
        event_input = torch.Tensor(event_signal).to(device)
        temporal_features = torch.Tensor(temporal_features).to(device)

        trend_input = torch.cat((trend_input, temporal_features), dim=3)
        event_input = torch.cat((event_input, temporal_features), dim=3)

        trend_input = trend_input.transpose(1, 3)
        event_input = event_input.transpose(1, 3)

        with torch.no_grad():
            prediction = model(trend_input, event_input).transpose(1, 3)

        predictions.append(prediction.squeeze(-1))

    predicted_output = torch.cat(predictions, dim=0)
    predicted_output = predicted_output[: ground_truth.size(0), ...]

    mae_values = []
    mape_values = []
    wmape_values = []
    rmse_values = []

    for horizon in range(12):
        prediction = scaler.inverse_transform(predicted_output[:, :, horizon])
        real = ground_truth[:, :, horizon]

        metrics = util.metric(prediction, real)

        log = (
            "Evaluate PGHA-Net on test data for horizon {:d}, "
            "Test MAE: {:.4f}, Test MAPE: {:.4f}, "
            "Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
        )

        print(
            log.format(
                horizon + 1,
                metrics[0],
                metrics[1],
                metrics[2],
                metrics[3]
            )
        )

        mae_values.append(metrics[0])
        mape_values.append(metrics[1])
        rmse_values.append(metrics[2])
        wmape_values.append(metrics[3])

    log = (
        "On average over 12 horizons, PGHA-Net Test MAE: {:.4f}, "
        "Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
    )

    print(
        log.format(
            np.mean(mae_values),
            np.mean(mape_values),
            np.mean(rmse_values),
            np.mean(wmape_values)
        )
    )


if __name__ == "__main__":
    main()
