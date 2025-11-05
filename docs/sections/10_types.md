# Data layers

The table below summarizes the available **data layers** in Imaging Server Kit.

In Python, you can use `help()` on a data layer object to access its detailed documentation.

| Type          | Description                                                                                                                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `sk.Image`    | An n-D image object represented as a Numpy array. Can validate array dimensionality.                                                |
| `sk.Mask`     | A segmentation mask as a Numpy array of integers. Can validates array dimensionality.                                               |
| `sk.Choice`   | A choice of `items` (available choices, rendered as a dropdown selector). Can be used to represent labels for classification.       |
| `sk.Float`    | A decimal value. Validates `min` and `max` values. `step` is used to specify incremental steps in UI layouts.                       |
| `sk.Integer`  | An integer value. Validates `min` and `max` values. `step` is used to specify incremental steps in UI layouts.                      |
| `sk.Bool`     | A boolean value. Represented as a checkbox in user interfaces.                                                                      |
| `sk.String`   | A string of text.                                                                                                                   |
| `sk.Points`   | A collection of point coordinates (n-D, can validate dimensionality).                                                               |
| `sk.Vectors`  | A collection of vectors (n-D, can validate dimensionality)                                                                          |
| `sk.Boxes `   | A collection of bounding boxes.                                                                                                     |
| `sk.Paths `   | A collection of paths, for example spline curves.                                                                                   |
| `sk.Tracks`   | Tracking data.                                                                                                                      |
| `sk.Notification` | A text notification (with levels `info`, `warning`, or `error`).                                                                |
| `sk.Null`     | Represents `None`, `NaN` or null values.                                                                                            |
