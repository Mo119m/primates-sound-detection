# Drop your data here

This `data/` folder is the **local drop-in workspace**. Clone the repo, copy your
audio into the folders below, and run
[`main_pipeline_notebooks/main_local.ipynb`](../main_pipeline_notebooks/main_local.ipynb) —
no path configuration needed. The notebook sets `PRIMATE_DATA_ROOT` to this
folder automatically.

The audio files themselves are **git-ignored** (only this README and the empty
folder markers are tracked), so dropping data here will not bloat your clone or
get committed by accident.

## Where each file goes

All clips are 2-second WAV files unless noted. Folder names must match exactly
(they are referenced in `src/config.py`).

### `species/` — labelled call clips (training positives)

| Folder | What goes in it |
|---|---|
| `CERNIC putty-nose 2s/` | Putty-nosed monkey *putty-nose* calls |
| `CERNIC hacks/` | Putty-nosed monkey *hack* calls |
| `CERNIC keks/` | Putty-nosed monkey *kek* calls |
| `CERNIC pyows/` | Putty-nosed monkey *pyow* calls |
| `CERNIC field_confirmed/` | *(optional)* Field-verified Cernic calls recovered during label audit |
| `Colobus guereza 2s windows/` | *Colobus guereza* roar clips |
| `Colobus_confuser/` | *(optional)* Hard negatives: forest sounds the model mistakes for *Colobus* |

The four `CERNIC *` call-type folders are pooled into a single **Cernic** class
at training time.

### `background/` — negative clips (5-second clips are fine here)

| Folder | What goes in it |
|---|---|
| `background noise Clips 5sec/` | Generic forest ambience / noise |
| `Cercocebus torquatus Clips 5s/` | Non-target species (mangabey) |
| `Pan troglodytes Clips 5sec/` | Non-target species (chimpanzee) |
| `wrong classified/` | Misc. confirmed non-target sounds |
| `field_fp_negatives/` | *(optional)* Field false positives mined as hard negatives |

### `long_audio/` and `field_recordings/`

- `long_audio/` — continuous recordings to run detection on (any length WAV).
- `field_recordings/` — IPA-station subfolders (e.g. `IPA1ST/`) for
  `scripts/run_detection_ipa.py`.

## Empty / optional folders

Folders marked *(optional)* can stay empty — the loader prints a warning and
skips a missing or empty folder, so the pipeline still runs. At minimum, put
clips in the `CERNIC *` folders, `Colobus guereza 2s windows/`, and a couple of
`background/` folders to train, plus one file in `long_audio/` to detect.

## Outputs

Trained models, detection CSVs, and visualizations are written to
`data/outputs/` (created automatically). That folder is also git-ignored.
