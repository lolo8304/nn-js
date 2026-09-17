# nn-js

Zero-dependency JavaScript/browser **inference** library for models saved by the Java `ch.lolo.nn` library (`JNN1`,
version 1).

## Browser

```js
import {Tensor, SequentialModel} from "./src/index.js";

const model = await SequentialModel.load("/models/mnist.nn");
const input = Tensor.of([[/* features */]]);
const output = model.predict(input);

console.log(output.toNestedArray());
console.log(model.predictClasses(input).toNestedArray());
```

A user-selected file works directly:

```js
const file = document.querySelector('input[type=file]').files[0];
const model = await SequentialModel.load(file);
```

`load()` accepts `File`/`Blob`, `ArrayBuffer`, typed arrays, or a URL. URL loading uses `fetch`, so normal
same-origin/CORS rules apply.

## Java compatibility

The reader consumes the exact binary format written by Java `ModelIO`: magic `JNN1`, version 1, layer definitions,
big-endian IEEE-754 doubles and tensor shapes. It supports all V1 persisted layers:

- Dense
- ReLU
- Sigmoid
- Tanh
- LeakyReLU
- Softmax
- Flatten
- Dropout

Dropout is intentionally a no-op during browser inference, matching Java `predict()`.

The optimizer/loss/training-state tail of a `.nn` file is not needed for inference and is safely ignored after the
layer/weight section.

## Run the included browser demo

```bash
npm run serve
# open http://localhost:8080
```

The demo automatically loads `examples/browser/mnist/mnist.nn` and 100 MNIST samples,
shows each digit, and compares its prediction with the expected label. Use **Next sample**
or **Random sample** to test more digits. You can also select another MNIST `.nn` model
or edit the normalized 784-pixel JSON batch.

Draw your own digit beside the sample and click **Detect**. The drawing is resized to
fit approximately 20×20 pixels while preserving its proportions, then centered by
intensity-weighted center of mass within 28×28 pixels. Black padding and bilinear
interpolation preserve grayscale values in 0–1. The existing JSON batch field opens
with the exact processed input sent to the model. Sample digits are left unchanged.

The bundled model is copied from `nn-java/mnist.nn`; the samples are the first 100
t10k images. Older models may have trained on t10k; the updated Java example uses
separate training, validation, and test data. Copy a newly trained model over the bundled
`examples/browser/mnist/mnist.nn` to update it.

## Test

```bash
npm test
```

No runtime dependencies and no bundler are required.
