

import os
import argparse

import torch
from torchvision import transforms
from torchmetrics import Accuracy

import prepare_data, train_funcs, build_model, utils


parser = argparse.ArgumentParser(description="Argparser for hyper-parameters")

parser.add_argument("--num_epochs", 
                     default=30, 
                     type=int, 
                     help="the number of epochs")

parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

parser.add_argument("--num_filters",
                    default=32,
                    type=int,
                    help="number of filters to use in convolution layers")

parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning-rate")

parser.add_argument("--train_dir",
                    default="data_food101/data/train",
                    type=str,
                    help="directory path of training data")

parser.add_argument("--test_dir",
                    default="data_food101/data/test",
                    type=str,
                    help="directory path of testing data")


args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
NUM_FILTERS = args.num_filters
LEARNING_RATE = args.learning_rate

print("[INFO] Setup - Epochs : {} | Batch_size : {} | Num_filters : {} | Learning_rate : {}".format(NUM_EPOCHS, 
                                                                                                    BATCH_SIZE, 
                                                                                                    NUM_FILTERS, 
                                                                                                    LEARNING_RATE))

train_dir = args.train_dir
test_dir = args.test_dir

print("[INFO] Training directory : {}".format(train_dir))
print("[INFO] Testing directory : {}".format(test_dir))


device = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), 
    transforms.ToTensor() 
])
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = prepare_data.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=BATCH_SIZE
)


model = build_model.CNNAugment_TinyVGG(num_channels=3, 
                                       num_filters=NUM_FILTERS, 
                                       num_classes=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss() # Softmax + CrossEntropy (built-in Softmax)

optimizer = torch.optim.Adam(params=model.parameters(), # "parameters" to optimize (apply gradient descent)
                             lr=LEARNING_RATE)                  # "l"earning "r"ate 
    
metric_accuracy = Accuracy().to(device) # from torchmetrics import Accuracy

    
train_funcs.train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  metric=metric_accuracy,
                  device=device,
                  epochs=NUM_EPOCHS)


utils.save_model(model=model,
                 target_dir="model_module/models",
                 model_name="CNNAugment_TinyVGG_modular.pth")
