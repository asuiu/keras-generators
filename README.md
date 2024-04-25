# keras-generators

Multi-dimensional/Multi-input/Multi-output Data preprocessing and Batch Generators for Tensorflow models

## Installation

You can find the library on PyPi [keras_generators](https://pypi.org/project/keras-generators/)

```bash
pip install keras_generators
```

## The reasons this library exists

This library solves several basic problems in area of data preprocessing (scaling and encoding) and batch generation for
Tensorflow models, for which there are no solution ( at the moment) in Tensorflow or other open-source libraries. There
are several functionalities in opensource libraries which solves some of the below problems, but only partially, and
there's no way to combine them into a single solution without performing some custom adaptation or extending their
functionalities. The libraries are [scikit-learn](https://scikit-learn.org/stable/)
, [TensorFlow](https://www.tensorflow.org/) and [tensorflow-datasets](https://pypi.org/project/tensorflow-datasets/)
and [tensorflow-transform](https://www.tensorflow.org/tfx/transform/get_started).

### Problem 1:

#### Batch generation from multi-variate Timeseries data with train/test/val split and z-score scaling

Imagine you have a timeseries data (like weather temperature, stock market prices) and you need to train a neural model
to predict the next value in the sequence based on the sequential input.
Basic operations you have to do:

##### - split the data into train/test/val sets

You can
use [TimeseriesGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator)
from Tensorflow which is able to ingest multi-variate timeseries data and produce batches of inputs and targets. It's
able to take time series parameters such as stride, length of history, etc., and produce batches for
training/validation. But, it's not able to generate multi-step `target` data, and work on train/test splits. You'll need
to split the data manually by taking care of lookback and target data lengths, with all the possible edge cases of
stride and sampling parameters.

##### - generate `target` data

You'll have to generate `target` data by yourself - there's no functionality in any of above library to extract it from
the ingested timeseries, and this might not be trivial, especially if you want to generate multi-step targets (like
predicting 3 data-points ahead).

##### - split the data into train/test/val

-you'll have to do it sets by yourself - and this is not trivial as just splitiing by index won't work, as you have
lookback to take care of, and you'll have to split the target data perfectly aligned with the input data.
The [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) is able to do that by
using [window](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window)
, [skip](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip)
and [take](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take)
and [batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) methods. It's able to align the targets (
if you'd have those somehow generated), but it's not able to perform re-shuffling of data at the end of the epoch, as
well as scaling/normalizing of the data. Although there
exists [tft.scale_to_z_score](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_z_score), it doesn't
let you save the coefficients and scale the data for the inference, thus rendering it unusable for production use.

##### - scaling/normalizing the data + encoding

- you'll have to perform encoding and scaling of the data by yourself, and take care of saving coefficients and
  parameters of encoding for the inference. There
  are [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  and [encoders](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.inverse_transform)
  which are able to perform de-normalizing of the target predictions for inference (inverse_transform), as well as save
  coefficients for normalizing the new input data for inference, but it will have to be saved separately for every input
  and output separately, i.e. if you have multi-input, or multi-output network, for every I/O layer you'll have to save
  the scalers separately and apply them before inference.

##### - re-shuffling after every epoch

- you'll need to perform re-shuffling of the data at the end of the epoch by yourself, so you need to create custom
  batch generator class to be used by TF trainer.

### Problem 2

#### Multi-input / multi-output data preprocessing and batch generation

Tensorflow models can have multiple inputs and multiple outputs. For example, a model can have 2 inputs (X1, X2) and 2
outputs (Y1, Y2). In this case, the data preprocessing and batch generation should be done in a way that the data is
scaled and encoded for each input and output separately, as well as split into train/test keeping the input and target
data aligned. This is not possible with the current Keras API or existing libraries.

### Solution to above problems

##### `keras_generators.generators.TimeseriesDataSource` is able to:

- split the data train/test/val split using `TimeseriesDataSource.split` and splitters from `keras_generators.splitters`
  module
- perform automatic encoding/scaling using `keras_generators.encoders.DataEncoder` instances as parameters
- decode/denormalize the predicted data
- generate multi-step target data using `TimeseriesTargetsParams` as parameter.

##### `keras_generators.generators.DataSet` is able to:

- aligned split of multi-input/multi-output data into train/test/val sets, and in the same time prform
  fit-encoding/scaling of train data and use the fit scalers and encoders for validation and test data.

##### `keras_generators.generators.XBatchGenerator` and `keras_generators.generators.XBatchGenerator` is able to:

- generate batches of data for inference (`XBatchGenerator`) and training (`XYBatchGenerator`) for
  multi-input/multi-output models
- perform re-shuffling of the data at the end of the epoch

All the above classes are used in a pipeline, and you can find an example of their usage in the example
model [here](https://github.com/asuiu/keras-generators/blob/master/examples/predict_stock_price.py#L111).

## Example

Generate multi-input/multi-step output Neural Network model.

Multiple Input: multi-variate timeseries + tabular data

Output: Multi-step timeseries target (predicting 3 data-points ahead), with stride 2, on second timeseries input(target_idx=1).

```python
# input_df - input Dataframe with multi-variate timeseries data
price_input_ds = TimeseriesDataSource(name="input", tensors=input_df.values, length=60,
                                      target_params=TimeseriesTargetsParams(delay=0, pred_len=3, stride=2, target_idx=1)
                                      )

# Z-score scale data (input & output)
encoded_input_ds = price_input_ds.encode([ScaleEncoder(StandardScaler())])

# get targets with delay of 1 and prediction length of 3
targets_ds = TargetTimeseriesDataSource.from_timeseries_datasource(encoded_input_ds, name="target")

# tabular_df - Dataframe with tabular data. We scale data here with MinMax scaler
tabular_ds = TensorDataSource(name="tabular", tensors=tabular_df.values).encode([ScaleEncoder(MinMaxScaler())])

# Get train/val/test generators for Keras
dataset = DataSet(input_sources={encoded_input_ds.name: encoded_input_ds, "tabular": tabular_ds},
                  target_sources={targets_ds.name: targets_ds})
train_ds, val_ds, test_ds = dataset.split(splitter=OrderedSplitter(train=0.6, val=0.2))
train_gen = XYBatchGenerator(train_ds.input_sources, train_ds.target_sources, batch_size=32)
val_gen = XYBatchGenerator(val_ds.input_sources, val_ds.target_sources, batch_size=1024)
test_gen = XYBatchGenerator(test_ds.input_sources, test_ds.target_sources, batch_size=1024)

# Train model
history = model.fit(train_gen, epochs=20, validation_data=val_gen, )

# Inference and de-normalize/decode the predictions
y_pred = model.predict(test_gen.get_X_generator())
res_ds = TensorDataSource(name="prediction", tensors=y_pred, encoders=targets_ds.get_encoders())
unscaled_y_pred = res_ds.decode()[:]
```

# Changelog

All notable changes to this project will be documented in this file.

## [1.4.5] - 2024-04-25
- explicit use of tf_keras in model_object.py 

## [1.4.4] - 2024-04-24
- removed serialization of the model in the callbacks
- Renames SerializableKerasObject -> SerializableCallback

## [1.4.3] - 2024-04-23
- fix callbacks to work on TF 2.16
- added integration tests for callbacks.py module
- using dill serializer instead of pickle for serialization of the Callbacks

## [1.4.2] - 2024-04-22
- Drops support <3.10 in setup.py
- Upgrade ModelParams to use pydantic>=2.*
- Proper serialization of Callbacks using cloudpickle + dependency on cloudpickle
- Added common SerializableKerasObject
- Use legacy Keras (2.*) instead of the new 3.0 by explicitly importing tf_keras lib
- Integration tests for Callback serialization

## [1.4.1] - 2024-04-22 ( Removed)

## [1.4.0] - 2024-04-22
### Added
- Upgrade to support TensorFlow 2.16, Python 3.12.

### Removed
- Drop support for Python versions < 3.10.
- Drop support for TensorFlow versions < 2.16.
- Drop support for pydantic 1.x.

## [1.3.2] - 2023-12-01
### Added
- Added `state_autoclear` option to `ModelObject` as a workaround to memory leaks in Keras; it automatically calls `K.clear_session()` once every N calls to predict/evaluate.

## [1.3.1] - 2023-05-30
### Added
- Added `XYWBatchGenerator` to handle sample weights.

## [1.3.0] - 2023-04-13
### Changed
- `ModelParams` now inherit `ImmutableExtModel` from pyxtension.
- Removed unused custom Models from common.

## [1.2.6] - 2023-03-08
### Changed
- Model directory name will now be prefixed with 'TS' instead of suffixed.

## [1.2.5] - 2023-03-08
### Added
- Added "reverse" option to `OrderedSplitter`.

## [1.2.4] - 2023-01-15
### Added
- Added `add_default_callbacks` parameter to `ModelObject.train()`.

## [1.2.3] - 2023-01-05
### Added
- Now accepts empty validation and test in data split.

## [1.2.2] - 2022-12-30
### Removed
- Removed fixed protobuf dependency due to TensorFlow upgrade.

## [1.2.1] - 2022-12-29
### Added
- Major improvements and bug fixes in `CompoundDataSource`.

## [1.2.0] - 2022-12-14
### Added
- Added `callbacks.py`, various data encoders, and unchain functionality for `CompoundDataSource`.
- Added `predict_raw`, `evaluate_raw` methods and `MetrickCheckpoint` as default callback to `SimpleModelObject`.
- Code reformat with black.

## [1.1.9] - 2022-09-25
### Added
- Added `ChainedDataEncoder` and `CompoundDataEncoder`.
- Fixed `CompoundDataSource` to use new encoders.

## [1.1.8] - 2022-09-25
### Changed
- Extended `TensorsDataSource.__getitem__` to accept numpy int indexing.

## [1.1.7] - 2022-09-25
### Fixed
- Bugfix for generators.

## [1.1.6] - 2022-09-25
### Added
- Added `CompoundDataSource`.

## [1.1.5] - 2022-09-24
### Added
- Added `DataSet.get_encoders()` and `split()` methods.
- Improved typing annotations.
- Added usage example to README.

## [1.1.4] - 2022-09-24
### Fixed
- Regression in `SimpleModelObject.from_model_dir()`.
- Adjusted example to save and load trained model to/from disk.

## [1.1.3] - 2022-09-24
### Added
- Added `TargetTimeseriesDataSource`, default layer names to `ModelParams`, and `DataSource.select_features()`.

## [1.1.2] - 2022-09-21
### Changed
- TimeseriesDataSource.get_targets() now returns `TensorDataSource`.
- Moved several classes to common.py.
- Updated README.md with the motivation of the library.

## [1.1.1] - 2022-09-21
### Added
- Added model abstractions (ModelParams & ModelObject).
- Added examples with model training using keras-generators.

## [1.1.0] - 2022-09-21
### Added
- Introduced data encoders, DataSource based generators, and adapted existing Splitters to new class architecture.

## [1.0.2] - 2022-09-02
### Fixed
- Fixed imports for compatibility with breaking changes in TensorFlow 2.9.

## [1.0.1] - 2022-09-01
### Initial
- First commit with initial functionality.

## [1.0.0] - 2022-09
