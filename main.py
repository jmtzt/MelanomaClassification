import os
import torch
import numpy as np
import pandas as pd
from sklearn import metrics

import torch.nn as nn
from torch.nn import functional as F

import albumentations
import pretrainedmodels

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

from apex import amp


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


def train(fold):
    print(f"Training fold #{fold}")
    training_data_path = "/mnt/Data/MelanomaClassification/input/kaggle/working/train224/"
    model_path = "/mnt/Data/MelanomaClassification/models/"
    df = pd.read_csv("/mnt/Data/MelanomaClassification/input/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png")
                    for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png")
                    for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"  # because the metric is AOC
    )

    # mixed precision training with apex
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", verbosity=0)

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            fp16=True
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device=device,
        )

        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(
            f"epoch={epoch} -- training loss={training_loss} -- valid loss={valid_loss}-- auc={auc}"
        )
        es(auc, model, os.path.join(model_path, f"model_fold_{fold}.bin"))
        if es.early_stop:
            print("Early stopping!")
            break


def predict(fold):
    print(f"Predicting fold #{fold}")
    test_data_path = "/mnt/Data/MelanomaClassification/input/kaggle/working/test224/"
    df = pd.read_csv("/mnt/Data/MelanomaClassification/input/test.csv")
    device = "cuda"
    model_path = os.path.join(
        "/mnt/Data/MelanomaClassification/models/", f"model_fold_{fold}.bin"
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".png") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    model = SEResNext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions


if __name__ == "__main__":
    # n_folds = 10
    # for i in range(n_folds):
    #     train(i)
    train(0)

    p0 = predict(0)
    # p1 = predict(1)
    # p2 = predict(2)
    # p3 = predict(3)
    # p4 = predict(4)
    # p5 = predict(5)
    # p6 = predict(6)
    # p7 = predict(7)
    # p8 = predict(8)
    # p9 = predict(9)

    # predictions = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 10

    predictions = p0

    sample = pd.read_csv(
        "/mnt/Data/MelanomaClassification/input/submission/sample_submission.csv"
    )
    sample.loc[:, "target"] = predictions
    sample.to_csv("submission.csv", index=False)
