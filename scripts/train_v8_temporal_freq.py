"""
Train V8: VGG19 backbone + time-frequency CRNN head (two-stage fine-tuning).

Builds on V7 with two improvements:
  1. Cleaned training data: human-verified Colobus and Cernic positive samples
     with mislabelled clips (bird/insect-only windows) removed.
  2. Time-frequency head (pooling='temporal_freq'): splits the VGG feature map
     into 4 frequency bands before temporal modelling. Each band gets its own
     Conv1D stream, then all merge for a cross-band Conv1D + BiLSTM. This lets
     the model explicitly learn WHERE in frequency energy sits (Colobus roar
     ~512Hz vs Cernic hack ~1kHz vs bird >2kHz) alongside WHEN it occurs.

Run (Colab)
-----------
    !cd /content/primates-sound-detection && \
      PRIMATE_DATA_ROOT="/content/drive/MyDrive/primates-sound-detection" \
      python scripts/train_v8_temporal_freq.py

The best model is checkpointed to {MODEL_SAVE_DIR}/best_model_v8.h5 throughout
both stages (monitored on val_accuracy).
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, train as train_mod, model as model_module

STAGE1_LR = 1e-4
STAGE2_LR = 1e-5
STAGE1_EPOCHS = config.EPOCHS
STAGE2_EPOCHS = config.EPOCHS
UNFREEZE_BLOCKS = 2
MODEL_FILENAME = "best_model_v8.h5"


def _callbacks(model_path, patience=config.PATIENCE):
    """Checkpoint best-on-val-accuracy + early stop + LR plateau, per stage."""
    return [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                        save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      min_delta=config.MIN_DELTA,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-7, verbose=1),
    ]


def main():
    print("=" * 70)
    print("  V8 training: VGG19 + time-frequency CRNN head")
    print("  (cleaned labels + frequency-band-aware temporal modelling)")
    print("=" * 70)

    # 1) Data -- uses cleaned positive samples (CERNIC putty-nose 2s,
    #    human-verified Colobus guereza 2s windows).
    X_train, X_val, y_train, y_val, class_names = train_mod.prepare_dataset()
    class_weights = train_mod.calculate_class_weights(y_train)

    model_path = os.path.join(config.MODEL_SAVE_DIR, MODEL_FILENAME)
    print(f"\n  Best model will be checkpointed to: {model_path}")

    # 2) Build the temporal_freq model (frozen base for stage 1).
    model = model_module.build_model(pooling="temporal_freq", freeze_base=True)

    # ---- Stage 1: train the new head only ----
    print("\n" + "=" * 70)
    print(f"  STAGE 1: frozen VGG base, train head only (LR={STAGE1_LR})")
    print("=" * 70)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=STAGE1_EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weights, callbacks=_callbacks(model_path), verbose=1,
    )

    # ---- Stage 2: unfreeze last 2 VGG blocks, fine-tune at a low LR ----
    print("\n" + "=" * 70)
    print(f"  STAGE 2: unfreeze last {UNFREEZE_BLOCKS} VGG blocks, "
          f"fine-tune (LR={STAGE2_LR})")
    print("=" * 70)
    model = model_module.unfreeze_base_model(model, num_blocks_to_unfreeze=UNFREEZE_BLOCKS)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=STAGE2_EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weights, callbacks=_callbacks(model_path), verbose=1,
    )

    # 3) Evaluate the best checkpoint.
    print("\n  Reloading best checkpoint for evaluation...")
    best = keras.models.load_model(model_path)
    train_mod.evaluate_model(best, X_val, y_val)

    print(f"\n  Done. V8 saved to {model_path}")
    print("  Next: run detection with this model on the held-out IPA19/20 and "
          "report -- do NOT tune anything further on the test stations.")


if __name__ == "__main__":
    main()
