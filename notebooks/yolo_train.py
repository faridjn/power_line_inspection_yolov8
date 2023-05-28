import os
import argparse
from ultralytics import YOLO
import torch

def train_yolo(data_yaml, imgsz, batch_size, epoch_size, optimizer, pretrained, lr0, resume, momentum, weight_decay, patience, trained_model):
    model = YOLO(trained_model)
    results = model.train(
        mode='detect',
        data=data_yaml,
        epochs=epoch_size,
        imgsz=imgsz,
        batch=batch_size,
        optimizer=optimizer,
        pretrained=pretrained,
        lr0=lr0,
        resume=resume,
        device=1,
        weight_decay=weight_decay,
        patience=patience
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--data', type=str, help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=768, help='Image size')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use a pretrained model')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimizer weight decay')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--trained_model', type=str, default='yolov8m.pt', help='Path to pretrained model')
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        epoch_size=args.epochs,
        optimizer=args.optimizer,
        pretrained=args.pretrained,
        lr0=args.lr0,
        resume=args.resume,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        trained_model=args.trained_model
    )

