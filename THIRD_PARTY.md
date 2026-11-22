# Third-party sources

The MORPH release includes the following source material. The original MORPH licenses and attribution are recorded in [Licensing](LICENSING.md).

## Tiny Shakespeare

William Shakespeare wrote the source plays. The training corpus is the Tiny Shakespeare selection distributed in Andrej Karpathy's *char-rnn*. The [upstream README](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/Readme.md) identifies MIT as the repository license. The text is reproduced byte for byte from the [pinned file](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt). The plays themselves are public-domain works in the United States. The training copy is `lm/data/input.txt`, SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

## Shakespeare's Sonnets

The external evaluation corpus is William Shakespeare's *Sonnets*, Project Gutenberg ebook 1041. The original edition and its full Project Gutenberg license are bundled in `lm_followup/data/source.txt`, SHA-256 `e3580b25135e2d89467dd4cef0ccdcdd5f50f8779247fc24612ef653d27dc9d5`. [Pinned GITenberg edition](https://github.com/GITenberg/Shakespeare-s-Sonnets_1041/blob/b6da8a3e73e7081cf71c57e930001d0b92a8f644/1041.txt)

> This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or online at [www.gutenberg.org](https://www.gutenberg.org/policy/license.html).

## Autoresearch

The language-model search adapts Andrej Karpathy's fixed-duration, validation-driven autoresearch procedure. MORPH provides its own MLX trainer, tasks and scoring code. The upstream README identifies the project as MIT licensed. [Pinned autoresearch program](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md) · [Upstream README](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md)

The MIT notice for material from the two Karpathy repositories is reproduced in [LICENSES/Karpathy-MIT.txt](LICENSES/Karpathy-MIT.txt).

## Figure fonts

The scientific figures embed subsets of DejaVu Sans. The original Bitstream and DejaVu notices accompany this release. [Bundled font notice](LICENSES/DejaVu.txt) · [DejaVu licenses](https://dejavu-fonts.github.io/License.html)
