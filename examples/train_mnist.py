import argparse

from aciq.mnist import train_model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a single MNIST model end-to-end")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=1170)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--eval-every", type=int, default=10)
  args = parser.parse_args()

  _, accuracy, train_losses, test_losses = train_model(
    seed=args.seed,
    steps=args.steps,
    lr=args.lr,
    batch_size=args.batch_size,
    eval_every=args.eval_every,
  )
  print(f"Final test accuracy: {accuracy:.4f}")
  print(f"Last logged train loss: {train_losses[-1]:.4f}")
  print(f"Last logged test loss:  {test_losses[-1]:.4f}")
