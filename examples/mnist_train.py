from aciq.mnist import train_model


if __name__ == "__main__":
  _, accuracy, train_losses, test_losses = train_model()
  print(f"Final test accuracy: {accuracy:.4f}")
  print(f"Last logged train loss: {train_losses[-1]:.4f}")
  print(f"Last logged test loss:  {test_losses[-1]:.4f}")
