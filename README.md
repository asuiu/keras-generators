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