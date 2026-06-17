"""
Train V12: VGG19 backbone + frequency-position CRNN head (two-stage fine-tuning).

This is the script that reproduces the published V12 model. It mirrors
``train_v8_temporal_freq.py`` exactly except for two settings:

  1. pooling = 'temporal_freqpos' (instead of 'temporal_freq'). This stamps an
     explicit frequency-position channel (FrequencyCoord / CoordConv) onto the
     VGG feature map before the four-band CRNN head, so every texture feature is
     tagged with the absolute frequency at which it occurs (see src/model.py).
  2. The best checkpoint is saved as best_model_v12.h5.

The high-frequency nuisance augmentation for Colobus is applied automatically
during data preparation: it is config-gated by COLOBUS_HF_AUG_CLASS /
COLOBUS_HF_CUTOFF_HZ / COLOBUS_HF_AUG_COUNT (= 2 by default), so no extra code
is needed here. Stage-2 fine-tuning unfreezes the last two blocks (block3,
block4) of the block4_conv4-truncated base.

Run (Colab)
-----------
    !cd /content/primates-sound-detection && \
      PRIMATE_DATA_ROOT="/content/drive/MyDrive/primates-sound-detection" \
      python scripts/train_v12.py

The best model is checkpointed to {MODEL_SAVE_DIR}/best_model_v12.h5 throughout
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
MODEL_FILENAME = "best_model_v12.h5"


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
    print("  V12 training: VGG19 + frequency-position CRNN head")
    print("  (frequency-position encoding + high-frequency nuisance augmentation)")
    print("=" * 70)

    # 1) Data -- human-verified clips; the Colobus high-frequency nuisance
    #    augmentation is applied automatically (config COLOBUS_HF_AUG_COUNT).
    X_train, X_val, y_train, y_val, class_names = train_mod.prepare_dataset()
    class_weights = train_mod.calculate_class_weights(y_train)

    model_path = os.path.join(config.MODEL_SAVE_DIR, MODEL_FILENAME)
    print(f"\n  Best model will be checkpointed to: {model_path}")

    # 2) Build the frequency-position model (frozen base for stage 1).
    model = model_module.build_model(pooling="temporal_freqpos", freeze_base=True)

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
    best = keras.models.load_model(
        model_path, custom_objects={"FrequencyCoord": model_module.FrequencyCoord})
    train_mod.evaluate_model(best, X_val, y_val)

    print(f"\n  Done. V12 saved to {model_path}")
    print("  Next: run detection with this model on the held-out IPA19/20 and "
          "report -- do NOT tune anything further on the test stations.")


if __name__ == "__main__":
    main()
