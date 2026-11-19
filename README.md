# mandelbrot-neon

[![ci](https://github.com/sushanth-dev/mandelbrot-neon/actions/workflows/ci.yml/badge.svg)](https://github.com/sushanth-dev/mandelbrot-neon/actions/workflows/ci.yml)

A Mandelbrot renderer in Rust with a hand-vectorized NEON inner loop (two `f64` pixels per instruction) next to a scalar loop, so the speedup is measured and not assumed. Standard library only, no dependencies. aarch64 only.

![Seahorse valley, 1920x1080](seahorse.png)

## Run it

```
cargo run --release -- render --cx -0.74364389 --cy 0.1318259 --zoom 400 --iter 3000 --out seahorse.ppm
cargo run --release -- bench
cargo test --release
```

Flags: `--cx --cy --zoom --width --height --iter --threads --scalar --out`. Output is a PPM file (`sips -s format png in.ppm --out out.png` converts it on macOS).

## Results

Apple M1 Pro, 8 threads, 1920x1080, `max_iter` 1000, default view. Median of 5 runs after one warm-up run:

| Config | Seconds | Giter/s | Speedup |
|---|---|---|---|
| scalar, 1 thread | 2.176 | 0.290 | 1.00x |
| neon, 1 thread | 1.098 | 0.575 | 1.98x |
| scalar, 8 threads | 0.348 | 1.811 | 6.25x |
| neon, 8 threads | 0.183 | 3.456 | 11.92x |

`bench` asserts that all four outputs are identical pixel for pixel, so the speedup is not bought with different maths. The unit tests check the same on odd widths (which exercise the scalar tail) and on a boundary-heavy deep view.

These numbers are from one machine. The first, unaveraged run of the same benchmark gave 4.75x for scalar on 8 threads and 10.48x for NEON on 8 threads, so expect a few tens of percent of run-to-run variation on the multi-threaded rows.

## How it works

- Each pixel iterates `z = z^2 + c` until `|z|^2 >= 4` or `max_iter`.
- The NEON loop keeps a sticky active mask per lane. A lane that escapes stops counting while its neighbour keeps going, and the loop exits when both lanes are done.
- Both loops use separate multiply and add, never fused multiply-add. That keeps rounding identical, which is why the iteration counts match exactly.
- Threads take one row at a time from a shared iterator. Rows near the set cost far more than rows far from it, so fixed bands would leave threads idle.
- Multi-threaded scaling is below the core count. The M1 Pro has 6 performance and 2 efficiency cores; I have not profiled whether that explains it.

## Limits

- `f64` precision limits how deep it can zoom. Pixels collapse into blocks past a point I have not measured.
- Colouring uses raw iteration counts, so bands are visible in some views.
- The only `unsafe` is the NEON intrinsics block in `neon_row`. NEON is part of the aarch64 baseline.
- Non-aarch64 targets fail to compile on purpose (`compile_error!`).

## License

MIT
