"""
Train V7: VGG19 backbone + temporal CRNN head (two-stage fine-tuning).

This is the principled fix for the field false-positive problem. The 'gap' /
'freq_bands' heads average away WHEN energy occurs, so a real Cernic call
(discrete low-frequency down-sweeping arcs with rhythm) and an insect pulse
train / continuous noise band collapse to the same feature vector -- which is
why the model scores pure noise at 0.99 while real calls sit at 0.61-0.90, and
why the 4657 mined field negatives could not help (they were not separable
under GAP). The temporal CRNN head (config.MODEL_POOLING='temporal') keeps the
time axis, so those negatives finally become learnable.

What it reuses (nothing is wasted)
----------------------------------
  * the VGG19 ImageNet backbone,
  * the exact mel-spectrogram preprocessing and 224x224 input,
  * ALL accumulated training data: species clips, multi-window Colobus, every
    background source incl. the mined field FPs.
Only the head changes, so this is a drop-in retrain.

Two-stage training (important: the CRNN head is randomly initialised)
  Stage 1: freeze the VGG base, train only the new head at LR 1e-4 so the head
           learns before any gradient reaches the pretrained conv weights.
  Stage 2: unfreeze the last 2 VGG blocks and fine-tune the whole thing at the
           lower LR 1e-5 so the conv features adapt to audio without being
           wrecked by large early gradients.

Run (Colab)
-----------
    !cd /content/primates-sound-detection && \
      PRIMATE_DATA_ROOT="/content/drive/MyDrive/primates-sound-detection" \
      python scripts/train_v7_temporal.py

The best model is checkpointed to {MODEL_SAVE_DIR}/best_model_v7.h5 throughout
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
MODEL_FILENAME = "best_model_v7.h5"


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
    print("  V7 training: VGG19 + temporal CRNN head")
    print("=" * 70)

    # 1) Data -- identical pipeline to every previous version (reuses all data).
    X_train, X_val, y_train, y_val, class_names = train_mod.prepare_dataset()
    class_weights = train_mod.calculate_class_weights(y_train)

    model_path = os.path.join(config.MODEL_SAVE_DIR, MODEL_FILENAME)
    print(f"\n  Best model will be checkpointed to: {model_path}")

    # 2) Build the temporal model (frozen base for stage 1).
    model = model_module.build_model(pooling="temporal", freeze_base=True)

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
    model.compile(  # must recompile after changing trainable flags
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

    # 3) Evaluate the best checkpoint (restore_best_weights already left the best
    #    stage-2 weights in `model`, but reload the checkpoint to be safe).
    print("\n  Reloading best checkpoint for evaluation...")
    best = keras.models.load_model(model_path)
    train_mod.evaluate_model(best, X_val, y_val)

    print(f"\n  Done. V7 saved to {model_path}")
    print("  Next: run detection with this model on the held-out IPA19/20 and "
          "report -- do NOT tune anything further on the test stations.")


if __name__ == "__main__":
    main()
