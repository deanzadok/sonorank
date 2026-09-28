"""Small utilities shared by the training entry points.

Reader's guide: ``SummaryLogger`` writes a compact, append-only record of phase
changes, epoch metrics, test results, and completion status. Full batch-level
progress remains in the normal console or launcher log.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class SummaryLogger:
    """Logger that writes a shortened summary log with only key metrics per epoch.

    This provides a concise overview of training progress without the verbose
    per-batch output from tqdm progress bars.

    Usage:
        summary = SummaryLogger(output_dir)
        summary.log_phase_start(1, "Frozen encoder")
        for epoch in range(num_epochs):
            train_metrics = train_epoch(...)
            val_metrics = evaluate(...)
            summary.log_epoch(epoch, num_epochs, train_metrics, val_metrics, is_best, best_auc)
        summary.log_test_results(test_metrics)
        summary.log_completion()
    """

    def __init__(self, output_dir: Path, filename: str = "summary.log"):
        """Initialize the summary logger.

        Args:
            output_dir: Directory to save the summary log file
            filename: Name of the summary log file (default: summary.log)
        """
        self.log_path = output_dir / filename
        self._init_log()

    def _init_log(self):
        """Initialize the summary log file with header."""
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Training Summary Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def log_config(self, config: Dict):
        """Log key configuration parameters.

        Args:
            config: Dictionary of configuration parameters
        """
        with open(self.log_path, 'a') as f:
            f.write("\nConfiguration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

    def log_phase_start(self, phase: int, description: str):
        """Log the start of a training phase.

        Args:
            phase: Phase number (e.g., 1, 2)
            description: Description of the phase (e.g., "Frozen encoder")
        """
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"PHASE {phase}: {description}\n")
            f.write(f"{'=' * 60}\n")

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict,
        is_best: bool = False,
        best_value: Optional[float] = None,
        metric_name: str = "AUC",
    ):
        """Log metrics for a single epoch.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs in this phase
            train_metrics: Dictionary of training metrics (loss, accuracy, auc, etc.)
            val_metrics: Dictionary of validation metrics
            is_best: Whether this epoch achieved the best validation score
            best_value: The best metric value if is_best=True
            metric_name: Name of the metric being tracked for "best" (default: AUC)
        """
        with open(self.log_path, 'a') as f:
            f.write(f"\nEpoch {epoch}/{total_epochs}\n")

            # Format train metrics
            train_str = f"  Train - Loss: {train_metrics.get('loss', 0):.4f}"
            if 'accuracy' in train_metrics:
                train_str += f", Acc: {train_metrics['accuracy']:.4f}"
            if 'precision' in train_metrics:
                train_str += f", Prec: {train_metrics['precision']:.4f}"
            if 'recall' in train_metrics:
                train_str += f", Rec: {train_metrics['recall']:.4f}"
            if 'auc' in train_metrics:
                train_str += f", AUC: {train_metrics['auc']:.4f}"
            if 'dice' in train_metrics:
                train_str += f", Dice: {train_metrics['dice']:.4f}"
            if 'iou' in train_metrics:
                train_str += f", IoU: {train_metrics['iou']:.4f}"
            if 'psnr' in train_metrics:
                train_str += f", PSNR: {train_metrics['psnr']:.2f}dB"
            if 'f1' in train_metrics:
                train_str += f", F1: {train_metrics['f1']:.4f}"
            if 'mae_deg' in train_metrics:
                train_str += f", MAE: {train_metrics['mae_deg']:.4f}°"
            if 'epe' in train_metrics:
                train_str += f", EPE: {train_metrics['epe']:.4f}"
            if 'motion_silhouette' in train_metrics:
                train_str += f", MotSil: {train_metrics['motion_silhouette']:.4f}"
            f.write(train_str + "\n")

            # Format val metrics
            val_str = f"  Val   - Loss: {val_metrics.get('loss', 0):.4f}"
            if 'accuracy' in val_metrics:
                val_str += f", Acc: {val_metrics['accuracy']:.4f}"
            if 'precision' in val_metrics:
                val_str += f", Prec: {val_metrics['precision']:.4f}"
            if 'recall' in val_metrics:
                val_str += f", Rec: {val_metrics['recall']:.4f}"
            if 'auc' in val_metrics:
                val_str += f", AUC: {val_metrics['auc']:.4f}"
            if 'dice' in val_metrics:
                val_str += f", Dice: {val_metrics['dice']:.4f}"
            if 'iou' in val_metrics:
                val_str += f", IoU: {val_metrics['iou']:.4f}"
            if 'psnr' in val_metrics:
                val_str += f", PSNR: {val_metrics['psnr']:.2f}dB"
            if 'f1' in val_metrics:
                val_str += f", F1: {val_metrics['f1']:.4f}"
            if 'mae_deg' in val_metrics:
                val_str += f", MAE: {val_metrics['mae_deg']:.4f}°"
            if 'epe' in val_metrics:
                val_str += f", EPE: {val_metrics['epe']:.4f}"
            if 'motion_silhouette' in val_metrics:
                val_str += f", MotSil: {val_metrics['motion_silhouette']:.4f}"
            f.write(val_str + "\n")

            if is_best and best_value is not None:
                f.write(f"  ** New best {metric_name}: {best_value:.4f} **\n")

    def log_message(self, message: str):
        """Log a custom message.

        Args:
            message: Message to log
        """
        with open(self.log_path, 'a') as f:
            f.write(f"\n{message}\n")

    def log_test_results(self, test_metrics: Dict):
        """Log final test set results.

        Args:
            test_metrics: Dictionary of test metrics
        """
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write("FINAL TEST RESULTS\n")
            f.write(f"{'=' * 60}\n")

            for key, value in test_metrics.items():
                # Skip internal tracking keys
                if key in ('num_batches', 'num_valid_batches'):
                    continue

                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                elif isinstance(value, (list, tuple)):
                    # Format list values nicely (e.g., per_joint_mae, per_joint_f1)
                    formatted_values = ', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in value)
                    f.write(f"  {key}: [{formatted_values}]\n")
                else:
                    f.write(f"  {key}: {value}\n")

    def log_completion(self, best_metric_name: str = "AUC", best_metric_value: Optional[float] = None):
        """Log training completion.

        Args:
            best_metric_name: Name of the best metric tracked
            best_metric_value: Final best metric value
        """
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 80}\n")
            if best_metric_value is not None:
                f.write(f"Best {best_metric_name}: {best_metric_value:.4f}\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n")
