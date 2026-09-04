from __future__ import annotations

import pandas as pd

from covid_xray.transfer_learning import TransferConfig, build_transfer_model
from covid_xray.transfer_learning.callbacks import MacroF1Callback
from covid_xray.transfer_learning.dataset import build_dataset

SMALL = TransferConfig(image_size=(32, 32), pretrained=False, batch_size=4, dense_units=8)


def test_macro_f1_callback_records_one_score_per_epoch(manifest: pd.DataFrame) -> None:
    dataset = build_dataset(manifest, SMALL, shuffle=False, augment=False)
    model = build_transfer_model(SMALL)
    callback = MacroF1Callback(dataset, name="val_macro_f1")

    model.fit(dataset, epochs=2, callbacks=[callback], verbose=0)

    assert len(callback.scores) == 2
    assert all(0.0 <= score <= 1.0 for score in callback.scores)
