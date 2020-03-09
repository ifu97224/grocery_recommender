import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

# imports the model in model.py by name
from model import BinaryClassifier


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, "model_info.pth")
    with open(model_info_path, "rb") as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(
        model_info["input_features"],
        model_info["hidden_dim"],
        model_info["output_dim"],
        model_info["momentum"],
        model_info["dropout_rate"],
        model_info["num_layers"],
    )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


# Gets training data in batches from the train_df.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train_df.csv"))
    train_x = train_data.drop("TARGET", axis=1)
    train_y = train_data[["TARGET"]]

    train_y = torch.from_numpy(train_y.values).float()
    train_x = torch.from_numpy(train_x.values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Gets test data in batches from the test_df.csv file
def _get_test_data_loader(batch_size, test_dir):
    print("Get test data loader.")

    test_data = pd.read_csv(os.path.join(test_dir, "test_df.csv"))
    test_x = test_data.drop("TARGET", axis=1)
    test_y = test_data[["TARGET"]]

    test_y = torch.from_numpy(test_y.values).float()
    test_x = torch.from_numpy(test_x.values).float()

    test_ds = torch.utils.data.TensorDataset(test_x, test_y)

    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    test_loader  - The PyTorch DataLoader that should be used during testing.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        correct = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)

            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()

        print("Epoch: {}, Train Loss: {}".format(epoch, total_loss / len(train_loader)))

        test_loss = 0
        correct = 0
        with torch.no_grad():

            for batch in test_loader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                model.eval()

                output = model(batch_x)
                loss = criterion(output, batch_y)
                test_loss += loss.data.item()
                batch_y = batch_y.type(torch.LongTensor)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(batch_y.view_as(pred)).sum().item()

                test_loss /= len(test_loader)
                test_acc = correct / len(test_loader.dataset)
                print("Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir", 
        type=str, 
        default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--data-dir-train", 
        type=str, 
        default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--data-dir-test", 
        type=str, 
        default=os.environ["SM_CHANNEL_TEST"]
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.99,
        metavar="M",
        help="Batch norm momentum (default: 0.99)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        metavar="D",
        help="Dropout rate (default: 0.2)",
    )

    # Model Parameters
    parser.add_argument(
        "--input_features",
        type=int,
        default=3,
        metavar="IN",
        help="number of input features to model (default: 3)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=10,
        metavar="H",
        help="hidden dim of model (default: 10)",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=2,
        metavar="OUT",
        help="output dim of model (default: 2)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        metavar="OUT",
        help="Number of hidden layers (default: 2)",
    )

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.train_batch_size, args.data_dir_train)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir_test)

    model = BinaryClassifier(
        args.input_features,
        args.hidden_dim,
        args.output_dim,
        args.momentum,
        args.dropout_rate,
        args.num_layers,
    ).to(device)

    ## TODO: Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, test_loader, args.epochs, criterion, optimizer, device)

    # Keep the keys of this dictionary as they are
    model_info_path = os.path.join(args.model_dir, "model_info.pth")
    with open(model_info_path, "wb") as f:
        model_info = {
            "input_features": args.input_features,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "momentum": args.momentum,
            "dropout_rate": args.dropout_rate,
            "num_layers": args.num_layers,
        }
        torch.save(model_info, f)

    model_path = os.path.join(args.model_dir, "model.pth")
    with open(model_path, "wb") as f:
        torch.save(model.cpu().state_dict(), f)
