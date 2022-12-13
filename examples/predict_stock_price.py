#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>
from glob import glob
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import yfinance as yf
from keras import layers, Model
from keras.optimizers import Adam

from pydantic import PositiveInt, conint
from sklearn.preprocessing import StandardScaler
from tsx import TS

from keras_generators.encoders import ScaleEncoder
from keras_generators.generators import TimeseriesDataSource, TimeseriesTargetsParams, DataSet, XYBatchGenerator, \
    TargetTimeseriesDataSource
from keras_generators.model_abstractions.model_object import SimpleModelObject
from keras_generators.model_abstractions.model_params import ModelParams
from keras_generators.splitters import OrderedSplitter


class LSTMModelParams(ModelParams):
    input_symbols: List[str]
    target_symbol_idx: int  # index in the above array of symbols
    start_ts: TS
    stop_ts: TS
    lookback: PositiveInt
    pred_len: PositiveInt = 1  # should be higher than n_true_trends
    delay: conint(ge=0) = 0
    seq_input_name: str = "seq"
    max_epochs: PositiveInt = 5
    lstm_dropout: float = None
    rnn_type: Literal["RNN", "LSTM", "GRU", "DENSE"] = "LSTM"
    use_cuDNN: bool = True
    lstm_rec_dropout: float = 0.1
    lstm_units: PositiveInt = 20
    bidirectional: bool = False


def simple_lstm_model_factory(mp: LSTMModelParams):
    """
    This is a factory function that creates a model object for the LSTM model.
    """
    input_tensor = layers.Input(shape=(mp.lookback, len(mp.input_symbols)), name=mp.seq_input_name)
    model_inputs = input_tensor

    if mp.rnn_type.upper() == "DENSE":
        d_l = layers.Flatten()(model_inputs)
    else:
        recurrent_args = dict(dropout=mp.lstm_dropout, kernel_initializer="glorot_normal", bias_initializer="zero")
        if mp.rnn_type.upper() == "LSTM":
            rnn_layer_type = layers.LSTM
            recurrent_args["recurrent_activation"] = "sigmoid"
        elif mp.rnn_type.upper() == "GRU":
            rnn_layer_type = layers.GRU
            recurrent_args["recurrent_activation"] = "sigmoid"
        elif mp.rnn_type.upper() == "RNN":
            rnn_layer_type = layers.SimpleRNN
        else:
            raise ValueError(f"Unknown rnn_type: {mp.rnn_type}")

        if mp.use_cuDNN:
            recurrent_args["activation"] = "tanh"
            recurrent_args["recurrent_dropout"] = 0
            recurrent_args["unroll"] = False
            recurrent_args["use_bias"] = True
        else:
            recurrent_args["activation"] = "selu"
            recurrent_args["recurrent_dropout"] = mp.lstm_rec_dropout

        rnn_layer = rnn_layer_type(mp.lstm_units)
        if mp.bidirectional:
            rnn1 = layers.Bidirectional(rnn_layer)(model_inputs)
        else:
            rnn1 = rnn_layer(model_inputs)
        d_l = rnn1

    output_tensor = layers.Dense(mp.out_dim, name=mp.target_name)(d_l)

    model_name = f"{['', 'bi'][mp.bidirectional]}{mp.rnn_type}{mp.lstm_units}"
    model = Model(inputs=model_inputs, outputs=output_tensor, name=model_name)
    metrics = ['mse', 'mae']
    loss = mp.loss
    model.compile(optimizer=Adam(learning_rate=mp.learning_rate), loss=loss, metrics=metrics)
    return model


if __name__ == '__main__':
    _mp = LSTMModelParams(
        input_symbols=["BRK-A"],
        target_symbol_idx=0,
        start_ts=TS("2010-01-01"), stop_ts=TS("2019-12-31"),
        lookback=60,
        batch_size=30,
        max_epochs=50,
        steps_per_epoch=None,
        lstm_units=20,

    )
    close_prices_by_symbol = {}
    for symbol in _mp.input_symbols:
        ticker_df = yf.download(symbol, start=_mp.start_ts.as_iso_date, end=_mp.stop_ts.as_iso_date, interval="1d")
        close_inputs = ticker_df['Close']
        close_prices_by_symbol[symbol] = close_inputs

    input_df = pd.DataFrame(close_prices_by_symbol)
    target_symbol = _mp.input_symbols[_mp.target_symbol_idx]

    price_input_ds = TimeseriesDataSource(name=_mp.seq_input_name, tensors=input_df.values, length=_mp.lookback,
                                          target_params=TimeseriesTargetsParams(delay=_mp.delay,
                                                                                pred_len=_mp.pred_len,
                                                                                stride=1,
                                                                                target_idx=_mp.target_symbol_idx)
                                          )
    encoded_input_ds = price_input_ds.fit_encode([ScaleEncoder(StandardScaler())])
    targets_ds = TargetTimeseriesDataSource.from_timeseries_datasource(encoded_input_ds, name=_mp.target_name)

    dataset = DataSet(input_sources={_mp.seq_input_name: encoded_input_ds}, target_sources={_mp.target_name: targets_ds})
    train_ds, val_ds, test_ds = dataset.split(splitter=OrderedSplitter(train=_mp.train_size_ratio, val=_mp.val_size_ratio))
    train_gen = XYBatchGenerator(train_ds.input_sources, train_ds.target_sources, batch_size=_mp.batch_size)
    val_gen = XYBatchGenerator(val_ds.input_sources, val_ds.target_sources, batch_size=_mp.val_batch_size)
    test_gen = XYBatchGenerator(test_ds.input_sources, test_ds.target_sources, batch_size=_mp.val_batch_size)

    model = simple_lstm_model_factory(_mp)
    model_object = SimpleModelObject(mp=_mp, model=model, encoders=train_ds.get_encoders())
    model_dir = SimpleModelObject.construct_model_dir(name=model.name, base_dir="model-data")
    history = model_object.train(train_gen, val_gen, device="/CPU:0", model_dir=model_dir)
    all_epochs = glob(str(model_dir / "*.hdf5"))
    last_hdf5 = all_epochs[-1]
    new_model_obj = SimpleModelObject.from_model_dir(Path(last_hdf5), model_params_cls=_mp.__class__, device="/CPU:0")
    metrics = new_model_obj.model.evaluate(test_gen)
    result_ds = new_model_obj.predict(test_gen.get_X_generator(), device="/CPU:0")
    true_pred = np.concatenate((test_gen.targets['target'].decode()[:], result_ds[:]), axis=1)
    print(true_pred)
    print(metrics)
