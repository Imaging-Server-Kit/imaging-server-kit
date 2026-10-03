# Data layers

The table below summarizes the available **data layers** in Imaging Server Kit.

In Python, you can use `help()` on a data layer object to access its detailed documentation.

| Layer         | Description                                                                                                                         | Data format |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `sk.Image`    | An n-D image: 2D or 3D arrays, optionally multichannel or RGB.                                                                      | `np.ndarray` |
| `sk.Mask`     | A segmentation mask: label image where integer values encode object classes, or instances when `merger` is set to `instances`.      | `np.ndarray` |
| `sk.Choice`   | A choice of `items` (rendered as a dropdown selector in user interfaces). Can be used to represent labels for classification.       | `str` |
| `sk.Float`    | A floating-point value.                                                                                                             | `float` |
| `sk.Integer`  | An integer value.                                                                                                                   | `int` |
| `sk.Bool`     | A boolean value (represented as a checkbox in user interfaces).                                                                     | `bool` |
| `sk.String`   | A string of text.                                                                                                                   | `str` |
| `sk.Points`   | A collection of point coordinates (2D, 3D).                                                                                         | `np.ndarray`, shape `(N, D)` |
| `sk.Vectors`  | A collection of vectors (2D, 3D).                                                                                                   | `np.ndarray`, shape `(N, 2, D)`; origin + displacement |
| `sk.Boxes `   | A collection of bounding boxes (2D, 3D). Can be oriented (OBB).                                                                     | `np.ndarray`, shape `(N, 4, D)`; box corners |
| `sk.Paths `   | A collection of paths, for example spline curves.                                                                                   | list of `(N, D)` arrays |
| `sk.Tracks`   | A collection of tracks (2D, 3D).                                                                                                    | `(N, D+1)` with columns `[ID, T, (Z), Y, X]` |
| `sk.Notification` | A text notification (`levels`: [`info`, `warning`, or `error`], printed to the terminal or displayed in user interfaces.).      | `str` |
| `sk.Null`     | Represents `None`, `NaN` or null values.                                                                                            | `None` |
| `sk.Any`      | Other kinds of data, for example custom classes, lists, or dictionaries. Not always serializable (= don't use with `serve`).        | - |
| `sk.Progress` | A progress bar to update, up to `max_val`.                                                                                          | `int`; completed steps |

## Features (measurements, classes) in object layers

The `Mask` (instances), `Points`, `Boxes`, `Vectors`, `Paths` and `Tracks` layers support passing `features`: metadata associated with individual objects, such as measurements or class labels.

<Briefly describe how features work and give examples for classes and measurements>