import os
import torch
from ultralytics import YOLO


def train_model():
    # Best hyperparameters based on previous experiments
    best_lr = 0.0002  # Fine-tuned learning rate
    best_batch_size = 16  # Optimized batch size3.
    epochs = 50  # Increased epochs for better training convergence

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO("yolov8s.pt")  # Using YOLOv8 small version for faster and better results

    print(f"\nTraining with optimized hyperparameters - Learning Rate: {best_lr}, Batch Size: {best_batch_size}")

    # Set environment variable to avoid OpenMP runtime issues
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Define the training save directory
    save_dir = f"./runs/train/optimized_lr_{best_lr}_bs_{best_batch_size}"

    # Train the model with optimized hyperparameters
    model.train(
        data="data.yaml",  # YAML configuration file
        epochs=epochs,  # Increased epochs
        batch=best_batch_size,  # Optimized batch size
        workers=8,  # Set number of workers based on CPU cores
        device=device,
        project=save_dir,
        lr0=best_lr,  # Optimized learning rate
        momentum=0.95,  # Higher momentum for better convergence
        weight_decay=0.0005,  # Regularization to prevent overfitting
        lrf=0.01,  # Reduced final learning rate for stability
        save_period=10,  # Save checkpoints every 10 epochs
        patience=5  # Early stopping if no improvement after 5 epochs
    )

    # Evaluate the model
    eval_metrics = model.val()
    fitness = eval_metrics.fitness
    print(f"Final Fitness Score: {fitness}")

    # Return the fitness score for logging or further analysis
    return fitness


if __name__ == '__main__':
    train_model()
