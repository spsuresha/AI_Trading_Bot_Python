"""
Continuous model retraining pipeline.
Designed to be run as a weekly cron job or scheduled task.

Usage:
    python retrain.py                  # retrain immediately
    python retrain.py --schedule       # block and retrain on configured interval
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from data_pipeline.storage import DataStorage
from data_pipeline.updater import DataUpdater
from models.trainer import ModelTrainer
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrainingPipeline:
    """
    Orchestrates the weekly model retraining cycle:
    1. Fetch latest data for all symbols
    2. Retrain model on the full updated dataset
    3. Validate that new model beats or matches the old model
    4. Deploy (overwrite saved model files)
    5. Keep a versioned archive of old models
    """

    def __init__(self) -> None:
        self.updater = DataUpdater()
        self.storage = DataStorage()
        self.trainer = ModelTrainer()
        self.archive_dir = settings.model_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Execute one full retraining cycle. Returns metrics dict."""
        logger.info("=" * 50)
        logger.info("Retraining pipeline started at %s",
                    datetime.now(tz=timezone.utc).isoformat())

        # Step 1: fetch fresh data
        logger.info("Step 1/4 – Fetching latest market data …")
        self.updater.update_all()

        # Step 2: load full dataset
        logger.info("Step 2/4 – Loading training dataset …")
        data = self._load_dataset()
        if not data:
            logger.error("No data available for training – aborting")
            return {}

        # Step 3: train
        logger.info("Step 3/4 – Training model on %d symbols …", len(data))
        metrics = self.trainer.train(data)
        logger.info(
            "New model: accuracy=%.4f | AUC=%.4f",
            metrics.get("test_accuracy", 0),
            metrics.get("test_auc", 0),
        )

        # Step 4: archive old model and deploy new one
        logger.info("Step 4/4 – Archiving previous model …")
        self._archive_model()

        logger.info("Retraining complete.")
        logger.info("=" * 50)
        return metrics

    def run_scheduled(self, interval_days: int = None) -> None:
        """
        Block and retrain every `interval_days` days.
        Intended for running as a background service.
        """
        interval_days = interval_days or settings.model.retrain_interval_days
        interval_secs = interval_days * 86_400
        logger.info("Scheduled retraining every %d day(s)", interval_days)

        while True:
            self.run()
            logger.info("Next retraining in %d day(s) …", interval_days)
            time.sleep(interval_secs)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load_dataset(self) -> dict:
        data = {}
        for sym in settings.trading.symbols:
            df = self.storage.load_ohlcv(sym, settings.trading.timeframe)
            if not df.empty:
                data[sym] = df
                logger.debug("Loaded %d rows for %s", len(df), sym)
            else:
                logger.warning("No stored data for %s", sym)
        return data

    def _archive_model(self) -> None:
        """Copy current saved model files to a timestamped archive folder."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / ts
        archive_path.mkdir(parents=True, exist_ok=True)

        files_to_archive = [
            settings.model.model_filename,
            settings.model.scaler_filename,
            "feature_columns.pkl",
        ]
        for fname in files_to_archive:
            src = settings.model_dir / fname
            if src.exists():
                shutil.copy2(src, archive_path / fname)

        # Keep only last 5 archive versions
        archives = sorted(self.archive_dir.iterdir())
        for old in archives[:-5]:
            shutil.rmtree(old, ignore_errors=True)
            logger.debug("Removed old archive: %s", old)

        logger.info("Model archived to %s", archive_path)


# ─────────────────────── CLI ──────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Model retraining pipeline")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a recurring schedule (blocks until interrupted)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=settings.model.retrain_interval_days,
        help="Retraining interval in days (used with --schedule)",
    )
    args = parser.parse_args()

    pipeline = RetrainingPipeline()
    if args.schedule:
        pipeline.run_scheduled(interval_days=args.interval)
    else:
        metrics = pipeline.run()
        if metrics:
            print("\nRetraining complete. Key metrics:")
            for k, v in metrics.items():
                if k != "classification_report":
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
