# Alignment and decoding attribution

Cross-attention/DTW alignment follows [OpenAI Whisper](https://github.com/openai/whisper/blob/main/whisper/timing.py).
Turbo head indices also appear in [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp).
No Python or C++ runtime is used.

The native C beam search follows the sequence-search approach in
[OpenAI Whisper decoding](https://github.com/openai/whisper/blob/main/whisper/decoding.py).
Token-entropy, confidence and temperature-fallback criteria follow
[whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/927cfce34f31707e17f2bff35c349632fb9e2c3a/src/whisper.cpp).
This implementation uses three beams, sequential sampled candidates, request-local
seeds and strict rejection when all candidates fail. It does not incorporate a
C++ runtime or claim identical decoding output.

MIT License

Copyright (c) 2022 OpenAI
Copyright (c) 2023-2026 The ggml authors

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
