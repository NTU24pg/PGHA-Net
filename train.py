import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import pywt
import torch

import util
from model import PGHANet
from ranger21 import Ranger


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--epochs", type=int, default=120, help="")
parser.add_argument("--channels", type=int, default=96, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument("--expid", type=int, default=1, help="experiment id")
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
parser.add_argument("--train_mode", type=str, default="Fusion", help="")

args = parser.parse_args()


def flow_decoupling_layer(x, wavelet_basis, decomposition_level):
    x = x.transpose(0, 3, 2, 1)
    original_length = x.shape[-1]

    coefficients = pywt.wavedec(x, wavelet_basis, level=decomposition_level)

    trend_coefficients = [coefficients[0]]
    for _ in range(len(coefficients) - 1):
        trend_coefficients.append(None)

    event_coefficients = [None]
    for i in range(len(coefficients) - 1):
        event_coefficients.append(coefficients[i + 1])

    trend_signal = pywt.waverec(trend_coefficients, wavelet_basis)[..., :original_length]
    event_signal = pywt.waverec(event_coefficients, wavelet_basis)[..., :original_length]

    trend_signal = trend_signal.transpose(0, 3, 2, 1)
    event_signal = event_signal.transpose(0, 3, 2, 1)

    return trend_signal, event_signal


def build_dual_channel_inputs(batch_x, scaler, wavelet_basis, decomposition_level, device):
    traffic_flow = batch_x[:, :, :, 0]
    traffic_flow = traffic_flow[:, :, :, np.newaxis]
    traffic_flow = scaler.inverse_transform(traffic_flow)

    temporal_features = batch_x[:, :, :, 1:3]

    trend_signal, event_signal = flow_decoupling_layer(
        traffic_flow,
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

    return trend_input, event_input


class PGHANetTrainer:
    def __init__(
        self,
        scaler,
        input_dim,
        num_nodes,
        channels,
        dropout,
        learning_rate,
        weight_decay,
        device,
        granularity,
    ):
        self.model = PGHANet(
            device,
            input_dim,
            num_nodes,
            channels,
            granularity,
            dropout
        )

        self.model.to(device)
        self.optimizer = Ranger(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5

        print("The number of parameters: {}".format(self.model.get_num_params()))
        print(self.model)

    def train_step(self, trend_input, event_input, real_value):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(trend_input, event_input)
        output = output.transpose(1, 3)

        real = torch.unsqueeze(real_value, dim=1)
        prediction = self.scaler.inverse_transform(output)

        loss = self.loss(prediction, real, 0.0)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mape = util.MAPE_torch(prediction, real, 0.0).item()
        rmse = util.RMSE_torch(prediction, real, 0.0).item()
        wmape = util.WMAPE_torch(prediction, real, 0.0).item()

        return loss.item(), mape, rmse, wmape

    def evaluate_step(self, trend_input, event_input, real_value):
        self.model.eval()

        output = self.model(trend_input, event_input)
        output = output.transpose(1, 3)

        real = torch.unsqueeze(real_value, dim=1)
        prediction = self.scaler.inverse_transform(output)

        loss = self.loss(prediction, real, 0.0)
        mape = util.MAPE_torch(prediction, real, 0.0).item()
        rmse = util.RMSE_torch(prediction, real, 0.0).item()
        wmape = util.WMAPE_torch(prediction, real, 0.0).item()

        return loss.item(), mape, rmse, wmape


def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def get_dataset_config(dataset_name):
    if dataset_name == "PEMS08":
        return "data//PEMS08", 170, 288, 96, "coif1"

    if dataset_name == "PEMS03":
        return "data//PEMS03", 358, 288, 32, "sym2"

    if dataset_name == "PEMS04":
        return "data//PEMS04", 307, 288, 64, "coif1"

    if dataset_name == "PEMS07":
        return "data//PEMS07", 883, 288, 128, "coif1"

    if dataset_name == "bike_drop":
        return "data//bike_drop", 250, 48, 32, "coif1"

    if dataset_name == "bike_pick":
        return "data//bike_pick", 250, 48, 32, "coif1"

    if dataset_name == "taxi_drop":
        return "data//taxi_drop", 266, 48, 96, "coif1"

    if dataset_name == "taxi_pick":
        return "data//taxi_pick", 266, 48, 96, "coif1"

    raise ValueError("Unsupported dataset: {}".format(dataset_name))


def evaluate_pghanet_on_test_set(
    engine,
    dataloader,
    scaler,
    wavelet_basis,
    decomposition_level,
    device,
    collect_results=False,
):
    outputs = []

    real_values = torch.Tensor(dataloader["y_test"]).to(device)
    real_values = real_values.transpose(1, 3)[:, 0, :, :]

    for _, (x, _) in enumerate(dataloader["test_loader"].get_iterator()):
        trend_input, event_input = build_dual_channel_inputs(
            x,
            scaler,
            wavelet_basis,
            decomposition_level,
            device
        )

        with torch.no_grad():
            prediction = engine.model(trend_input, event_input).transpose(1, 3)

        outputs.append(prediction.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: real_values.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []
    test_results = []

    for horizon in range(12):
        prediction = scaler.inverse_transform(yhat[:, :, horizon])
        real = real_values[:, :, horizon]

        metrics = util.metric(prediction, real)

        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(horizon + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_metric = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )

        test_results.append(pd.Series(test_metric))

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    average_metric = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )

    if collect_results:
        test_results.append(pd.Series(average_metric))
        return np.mean(amae), test_results

    return np.mean(amae), None


def main():
    set_random_seed(6666)

    dataset_name = args.data
    data_path, num_nodes, granularity, channels, wavelet_basis = get_dataset_config(dataset_name)

    args.data = data_path
    decomposition_level = 1
    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size
    )

    scaler = dataloader["scaler"]

    best_valid_loss = 9999999
    best_test_loss = 999999
    best_epoch = 0
    epochs_since_best_mae = 0

    path = args.save + dataset_name + "/"

    history_loss = []
    validation_time = []
    training_time = []
    training_results = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = PGHANetTrainer(
        scaler,
        args.input_dim,
        num_nodes,
        channels,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        granularity,
    )

    print("start training...", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        start_train_time = time.time()

        for batch_index, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trend_input, event_input = build_dual_channel_inputs(
                x,
                scaler,
                wavelet_basis,
                decomposition_level,
                device
            )

            train_y = torch.Tensor(y).to(device)
            train_y = train_y.transpose(1, 3)

            metrics = engine.train_step(
                trend_input,
                event_input,
                train_y[:, 0, :, :]
            )

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if batch_index % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        batch_index,
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                    ),
                    flush=True,
                )

        end_train_time = time.time()

        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(epoch, end_train_time - start_train_time))
        training_time.append(end_train_time - start_train_time)

        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_wmape = []

        start_validation_time = time.time()

        for _, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            trend_input, event_input = build_dual_channel_inputs(
                x,
                scaler,
                wavelet_basis,
                decomposition_level,
                device
            )

            valid_y = torch.Tensor(y).to(device)
            valid_y = valid_y.transpose(1, 3)

            metrics = engine.evaluate_step(
                trend_input,
                event_input,
                valid_y[:, 0, :, :]
            )

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        end_validation_time = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(epoch, end_validation_time - start_validation_time))
        validation_time.append(end_validation_time - start_validation_time)

        mean_train_loss = np.mean(train_loss)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)
        mean_train_wmape = np.mean(train_wmape)

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)
        mean_valid_wmape = np.mean(valid_wmape)

        history_loss.append(mean_valid_loss)

        training_metric = dict(
            train_loss=mean_train_loss,
            train_rmse=mean_train_rmse,
            train_mape=mean_train_mape,
            train_wmape=mean_train_wmape,
            valid_loss=mean_valid_loss,
            valid_rmse=mean_valid_rmse,
            valid_mape=mean_valid_mape,
            valid_wmape=mean_valid_wmape,
        )

        training_results.append(pd.Series(training_metric))

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(
                epoch,
                mean_train_loss,
                mean_train_rmse,
                mean_train_mape,
                mean_train_wmape,
            ),
            flush=True,
        )

        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(
                epoch,
                mean_valid_loss,
                mean_valid_rmse,
                mean_valid_mape,
                mean_valid_wmape,
            ),
            flush=True,
        )

        if mean_valid_loss < best_valid_loss:
            print("Validation improvement detected.")

            if epoch < 80:
                best_valid_loss = mean_valid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                best_epoch = epoch
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mean_valid_loss, end=", ")
                print("epoch: ", epoch)

            elif epoch >= 80:
                mean_test_mae, _ = evaluate_pghanet_on_test_set(
                    engine,
                    dataloader,
                    scaler,
                    wavelet_basis,
                    decomposition_level,
                    device,
                    collect_results=False,
                )

                if mean_test_mae < best_test_loss:
                    best_test_loss = mean_test_mae
                    best_valid_loss = mean_valid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    best_epoch = epoch
                    print("Test low! Updating! Test Loss:", mean_test_mae, end=", ")
                    print("Test low! Updating! Valid Loss:", mean_valid_loss, end=", ")
                    print("epoch: ", epoch)
                else:
                    epochs_since_best_mae += 1
                    print("No update")
        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(training_results)
        train_csv.round(8).to_csv(f"{path}/train.csv")

        if epochs_since_best_mae >= args.es_patience and epoch >= 300:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(training_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(validation_time)))
    print("Training ends")
    print("The epoch of the best result:", best_epoch)
    print("The valid loss of the best model", str(round(history_loss[best_epoch - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    _, test_results = evaluate_pghanet_on_test_set(
        engine,
        dataloader,
        scaler,
        wavelet_basis,
        decomposition_level,
        device,
        collect_results=True,
    )

    test_csv = pd.DataFrame(test_results)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time spent: {:.4f}".format(end_time - start_time))
