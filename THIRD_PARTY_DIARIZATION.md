# Diarization attribution and licenses

The C diarization implementation is an independent port of the inference
algorithms used by `pyannote/speaker-diarization-community-1`. It is not an
official pyannote implementation. Upstream Python source was consulted as a
specification; no Python source, interpreter, or runtime is included or executed.

## Source attribution

- [pyannote.audio 4.0.0](https://github.com/pyannote/pyannote-audio/tree/a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a):
  PyanNet, SincNet, statistics pooling, embedding extraction, clustering,
  and diarization aggregation. Copyright (c) 2019- CNRS;
  Copyright (c) 2020 CNRS; Copyright (c) 2020- CNRS;
  Copyright (c) 2021-2025 CNRS; Copyright (c) 2023 CNRS;
  Copyright (c) 2025- pyannoteAI. MIT terms below.
- [asteroid-filterbanks](https://github.com/asteroid-team/asteroid-filterbanks):
  parameterized even/odd sinc filter construction. Copyright (c) 2019
  Pariente Manuel. MIT terms below.
- [WeSpeaker ResNet](https://github.com/pyannote/pyannote-audio/blob/a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a/src/pyannote/audio/models/embedding/wespeaker/resnet.py):
  Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com),
  2022 Zhengyang Chen (chenzhengyang117@gmail.com),
  2023 Bing Han (hanbing97@sjtu.edu.cn), 2023 CNRS.
  Apache License 2.0.
- [VBx](https://github.com/pyannote/pyannote-audio/blob/a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a/src/pyannote/audio/utils/vbx.py):
  Apache License 2.0. Original algorithm by L. Burget (2021), with adaptations
  by P. Pálka (2025) and H. Bredin (2025), derived from
  [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx/blob/e39af548bb41143a7136d08310765746192e34da/VBx/VB_diarization.py).

`src/diarization/network.c` and `src/diarization/cluster.c` are distributed under
the [Apache License 2.0](licenses/Apache-2.0.txt). Modifications include C-only
execution, direct checkpoint loading, bounded single-window workspaces, CPU
intrinsics, shared embedding feature computation, and a PLDA transform that
uses the supplied diagonalization rather than an external eigensolver.
Other newly authored C files use the repository's MIT license.

## Model weights

The [Community-1 model](https://huggingface.co/pyannote/speaker-diarization-community-1)
is provided by pyannote under CC BY 4.0, with access conditions on Hugging Face.
Weights are not included in this repository. The C port does not change the
weights. Users must obtain authorized access and comply with the model license.

Model references: Alexis Plaquet and Hervé Bredin, *Powerset multi-class cross
entropy loss for neural speaker diarization*, INTERSPEECH 2023; Hongji Wang et al.,
*Wespeaker: A research and production oriented speaker embedding learning
toolkit*, ICASSP 2023; Federico Landini et al., *Bayesian HMM clustering of
x-vector sequences (VBx) in speaker diarization: theory, implementation and
analysis on standard tasks*, Computer Speech & Language, 2022.

## MIT permission notice

The copyright notices listed above apply to their respective upstream works.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
