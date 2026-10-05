# Algorithms

Create algorithms, combine them into collections, and run them.

::: imaging_server_kit.algorithm

::: imaging_server_kit.Algorithm
    options:
      members: false

::: imaging_server_kit.combine

::: imaging_server_kit.MultiAlgorithm
    options:
      members: false

## Shared interface

Algorithms, collections, and [clients](remote.md#imaging_server_kit.Client) implement the methods below.

::: imaging_server_kit.AlgorithmRunner
    options:
      members:
        - run
        - get_sample
        - get_n_samples
        - info
        - get_parameters
        - is_tileable
        - get_signature_params
        - run_generator

## Shortcuts

::: imaging_server_kit.run

::: imaging_server_kit.info
