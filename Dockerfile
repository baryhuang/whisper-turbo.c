# Benchmark-only image. whisper.cpp is an external comparison binary, not a
# dependency of the native C runtime. Neither build path runs Python.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make cmake git ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/whisper.cpp.git /reference && cd /reference && git checkout c44b60b8053bbf2a5c1e014f11323fb3f2485177 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DGGML_CUDA=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF && cmake --build build -j8
WORKDIR /work
COPY Makefile ./
COPY src ./src
COPY tools/import_ggml.c ./tools/import_ggml.c
COPY benchmarks ./benchmarks
RUN make -j8 all x86-tools OPENMP=-fopenmp

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc make libgomp1 libstdc++6 ca-certificates curl time util-linux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /work/build/ /app/
COPY --from=build /reference/build/ /reference/build/
COPY --from=build /reference/samples/jfk.wav /app/jfk.wav
ENV PORT=8080 OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
EXPOSE 8080
CMD ["/app/bench-health"]
