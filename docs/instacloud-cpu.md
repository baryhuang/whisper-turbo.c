# InstaCloud CPU inspection

Observed on 2026-09-09 using the InstaCloud CLI against `audio-hub`, branch
`main`. Configuration came from `services list` and `compute limits`; CPU
features came from `compute exec SERVICE -- lscpu` on each worker.

| Service | Provider | Configured resources | Observed processor | SIMD features |
| --- | --- | --- | --- | --- |
| transcribe-worker | fly | 8 shared vCPUs, 8192 MiB | AMD EPYC, family 25 model 1 | AVX2, FMA, F16C; no AVX-512 flag |
| transcribe-worker-2 | insta-compute | 8 shared vCPUs, 8192 MiB | Intel Xeon 6975P-C | AVX2, FMA, AVX-512, VNNI, BF16, FP16 |
| transcribe-worker-3 | insta-compute | 8 shared vCPUs, 8192 MiB | Intel Xeon 6975P-C | AVX2, FMA, AVX-512, VNNI, BF16, FP16 |

All three are x86-64, little endian, in `us-west`, with one configured machine
each and `always_on=false`. The stopped `bench-whisper` service is also
configured for 8 shared vCPUs / 8192 MiB; it was not started or inspected.

The AMD guest exposes 8 online CPUs. Both Intel guests expose 9 online CPUs.
On worker-2, `cpu.max` was `max 100000`, `cpuset.cpus.effective` was `0-8`,
and `memory.max` was `max`. `/proc/meminfo` reported 10410708 KiB total.
These guest-visible values differ from the platform configuration and do not
establish a higher resource entitlement. Use the configured eight-thread
budget initially; measure thread scaling before changing it.

Implementation direction: retain the generic C fallback, add x86 AVX2/FMA
kernels, and select AVX-512 kernels at runtime on supporting machines.
Check CPU and OS vector-state support. Do not compile every deployment with
unconditional AVX-512 or `-march=native` from a different build machine.
There is no A113X/ARM NEON requirement for these observed instances.

No compute resource limits, images, or deployments were changed. The CLI
may wake idle instances while running inspection commands. These are point-in-time
observations; recheck CPU flags after placement or provider changes.
