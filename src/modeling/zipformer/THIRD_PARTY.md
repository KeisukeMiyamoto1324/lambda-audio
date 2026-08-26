# Japanese Zipformer

The source files in this directory are derived from
`reazon-research/japanese-zipformer-base-k2` at revision
`b45effd42d558a9271bb6fb3a70cbeee17c7470c`.

Source: https://huggingface.co/reazon-research/japanese-zipformer-base-k2

License: Apache License 2.0

Local changes for Transformers 5 compatibility:

- `ZipformerModel.__init__` calls `self.post_init()`.
- Meta-device positional encodings are regenerated on the input device.
- The configuration defines `initializer_range` for weight initialization.
- Missing padding masks are created from the encoded feature shape.
- Legacy helpers were removed and the remaining functions were fully typed.
