//! Mandelbrot escape-time renderer: a scalar loop and a hand-vectorized NEON loop
//! (two f64 pixels per instruction), each optionally multi-threaded.
//! std only. Both loops use separate multiply and add (no fused multiply-add),
//! so they produce bit-identical iteration counts; `bench` and the tests check that.

#[cfg(not(target_arch = "aarch64"))]
compile_error!("the NEON path needs an aarch64 target");

use std::arch::aarch64::*;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy)]
struct View {
    x_min: f64,
    y_max: f64,
    step: f64,
    width: usize,
    height: usize,
    max_iter: u32,
}

impl View {
    fn new(cx: f64, cy: f64, zoom: f64, width: usize, height: usize, max_iter: u32) -> View {
        let step = 3.0 / zoom / width as f64;
        View {
            x_min: cx - step * width as f64 / 2.0,
            y_max: cy + step * height as f64 / 2.0,
            step,
            width,
            height,
            max_iter,
        }
    }
    fn cr(&self, x: usize) -> f64 {
        self.x_min + x as f64 * self.step
    }
    fn ci(&self, y: usize) -> f64 {
        self.y_max - y as f64 * self.step
    }
}

type RowFn = fn(&View, usize, &mut [u32]);

fn scalar_pixel(cr: f64, ci: f64, max_iter: u32) -> u32 {
    let (mut zr, mut zi) = (0.0f64, 0.0f64);
    for n in 0..max_iter {
        let (zr2, zi2) = (zr * zr, zi * zi);
        if zr2 + zi2 >= 4.0 {
            return n;
        }
        zi = 2.0 * (zr * zi) + ci;
        zr = (zr2 - zi2) + cr;
    }
    max_iter
}

fn scalar_row(v: &View, y: usize, out: &mut [u32]) {
    let ci = v.ci(y);
    for (x, px) in out.iter_mut().enumerate() {
        *px = scalar_pixel(v.cr(x), ci, v.max_iter);
    }
}

fn neon_row(v: &View, y: usize, out: &mut [u32]) {
    let ci = v.ci(y);
    let mut x = 0;
    // SAFETY: NEON is part of the aarch64 baseline; loads read a 2-element stack array.
    unsafe {
        let civ = vdupq_n_f64(ci);
        let four = vdupq_n_f64(4.0);
        let two = vdupq_n_f64(2.0);
        while x + 2 <= v.width {
            let crs = [v.cr(x), v.cr(x + 1)];
            let cr = vld1q_f64(crs.as_ptr());
            let (mut zr, mut zi) = (vdupq_n_f64(0.0), vdupq_n_f64(0.0));
            let mut active = vdupq_n_u64(u64::MAX);
            let mut count = vdupq_n_u64(0);
            for _ in 0..v.max_iter {
                let zr2 = vmulq_f64(zr, zr);
                let zi2 = vmulq_f64(zi, zi);
                // sticky mask: a lane that escaped never counts again
                active = vandq_u64(active, vcltq_f64(vaddq_f64(zr2, zi2), four));
                if vmaxvq_u32(vreinterpretq_u32_u64(active)) == 0 {
                    break;
                }
                count = vsubq_u64(count, active); // active is all ones (-1), so this adds 1
                zi = vaddq_f64(vmulq_f64(two, vmulq_f64(zr, zi)), civ);
                zr = vaddq_f64(vsubq_f64(zr2, zi2), cr);
            }
            out[x] = vgetq_lane_u64::<0>(count) as u32;
            out[x + 1] = vgetq_lane_u64::<1>(count) as u32;
            x += 2;
        }
    }
    for (i, px) in out.iter_mut().enumerate().skip(x) {
        *px = scalar_pixel(v.cr(i), ci, v.max_iter);
    }
}

/// Rows are handed out one at a time so threads stay balanced: rows near the set cost far more.
fn render(v: &View, threads: usize, row: RowFn) -> Vec<u32> {
    let mut buf = vec![0u32; v.width * v.height];
    let rows = Mutex::new(buf.chunks_mut(v.width).enumerate());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let next = rows.lock().unwrap().next();
                match next {
                    Some((y, out)) => row(v, y, out),
                    None => break,
                }
            });
        }
    });
    drop(rows);
    buf
}

fn color(n: u32, max_iter: u32) -> [u8; 3] {
    if n >= max_iter {
        return [0, 0, 0];
    }
    let t = (n as f64).sqrt() * 0.35;
    let ch = |phase: f64| (127.5 + 127.5 * (t + phase).cos()) as u8;
    [ch(0.0), ch(0.9), ch(2.4)]
}

fn write_ppm(path: &str, v: &View, counts: &[u32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{} {}\n255\n", v.width, v.height)?;
    for &n in counts {
        f.write_all(&color(n, v.max_iter))?;
    }
    f.flush()
}

struct Args {
    mode: String,
    cx: f64,
    cy: f64,
    zoom: f64,
    width: usize,
    height: usize,
    max_iter: u32,
    threads: usize,
    simd: bool,
    out: String,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        mode: "render".into(),
        cx: -0.5,
        cy: 0.0,
        zoom: 1.0,
        width: 1920,
        height: 1080,
        max_iter: 1000,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        simd: true,
        out: "mandelbrot.ppm".into(),
    };
    let mut it = std::env::args().skip(1);
    let mut first = true;
    while let Some(arg) = it.next() {
        if first && !arg.starts_with("--") {
            a.mode = arg;
            first = false;
            continue;
        }
        first = false;
        if arg == "--scalar" {
            a.simd = false;
            continue;
        }
        let val = it.next().ok_or(format!("{arg} needs a value"))?;
        fn num<T: std::str::FromStr>(arg: &str, val: &str) -> Result<T, String> {
            val.parse().map_err(|_| format!("bad value for {arg}: {val}"))
        }
        match arg.as_str() {
            "--cx" => a.cx = num(&arg, &val)?,
            "--cy" => a.cy = num(&arg, &val)?,
            "--zoom" => a.zoom = num(&arg, &val)?,
            "--width" => a.width = num(&arg, &val)?,
            "--height" => a.height = num(&arg, &val)?,
            "--iter" => a.max_iter = num(&arg, &val)?,
            "--threads" => a.threads = num(&arg, &val)?,
            "--out" => a.out = val,
            _ => return Err(format!("unknown flag {arg}")),
        }
    }
    if !(a.cx.is_finite() && a.cy.is_finite() && a.zoom.is_finite() && a.zoom > 0.0) {
        return Err("cx and cy must be finite and zoom must be finite and positive".into());
    }
    if a.width == 0 || a.height == 0 || a.threads == 0 || a.max_iter == 0 {
        return Err("width, height, threads and iter must be positive".into());
    }
    if a.mode != "render" && a.mode != "bench" {
        return Err(format!("unknown mode {}; use render or bench", a.mode));
    }
    Ok(a)
}

fn timed(v: &View, threads: usize, row: RowFn) -> (Vec<u32>, f64) {
    let t = Instant::now();
    let buf = render(v, threads, row);
    (buf, t.elapsed().as_secs_f64())
}

fn main() {
    let a = parse_args().unwrap_or_else(|e| {
        eprintln!("error: {e}\nusage: mandelbrot-neon [render|bench] [--cx F --cy F --zoom F --width N --height N --iter N --threads N --scalar --out FILE]");
        std::process::exit(2);
    });
    let v = View::new(a.cx, a.cy, a.zoom, a.width, a.height, a.max_iter);

    if a.mode == "render" {
        let row: RowFn = if a.simd { neon_row } else { scalar_row };
        let (counts, secs) = timed(&v, a.threads, row);
        write_ppm(&a.out, &v, &counts).expect("write failed");
        println!("wrote {} ({}x{}) in {:.3}s", a.out, v.width, v.height, secs);
        return;
    }

    const RUNS: usize = 5;
    let configs: [(&str, usize, RowFn); 4] = [
        ("scalar, 1 thread", 1, scalar_row),
        ("neon, 1 thread", 1, neon_row),
        ("scalar, all threads", a.threads, scalar_row),
        ("neon, all threads", a.threads, neon_row),
    ];
    let mut reference: Option<Vec<u32>> = None;
    let mut base = 0.0;
    println!("{}x{}, max_iter {}, {} threads, median of {RUNS} runs", v.width, v.height, v.max_iter, a.threads);
    println!("{:<22} {:>9} {:>14} {:>9}", "config", "seconds", "Giter/s", "speedup");
    for (i, (name, threads, row)) in configs.iter().enumerate() {
        let (buf, _) = timed(&v, *threads, *row); // warm-up, also the output compared below
        let total: u64 = buf.iter().map(|&n| n as u64).sum();
        let mut times: Vec<f64> = (0..RUNS).map(|_| timed(&v, *threads, *row).1).collect();
        times.sort_by(f64::total_cmp);
        let secs = times[RUNS / 2];
        if i == 0 {
            base = secs;
        }
        match &reference {
            None => reference = Some(buf),
            Some(r) => assert!(*r == buf, "{name} output differs from the scalar reference"),
        }
        println!("{:<22} {:>9.3} {:>14.3} {:>8.2}x", name, secs, total as f64 / secs / 1e9, base / secs);
    }
    println!("all four outputs are identical, pixel for pixel");
}

#[cfg(test)]
mod tests {
    use super::*;

    // Odd width exercises the scalar tail; a deep, boundary-heavy view exercises lanes that finish at different times.
    #[test]
    fn neon_matches_scalar_pixel_for_pixel() {
        for (cx, cy, zoom, w, h) in [
            (-0.5, 0.0, 1.0, 97, 53),
            (-0.74364389, 0.1318259, 5000.0, 64, 64),
            (0.0, 0.0, 0.5, 31, 31),
        ] {
            let v = View::new(cx, cy, zoom, w, h, 500);
            assert_eq!(render(&v, 1, scalar_row), render(&v, 1, neon_row));
            assert_eq!(render(&v, 1, scalar_row), render(&v, 4, neon_row));
        }
    }

    #[test]
    fn known_points() {
        assert_eq!(scalar_pixel(0.0, 0.0, 100), 100); // inside the set
        assert_eq!(scalar_pixel(2.0, 2.0, 100), 1); // |z1|^2 = 8 escapes at the second check
        assert_eq!(scalar_pixel(-1.0, 0.0, 100), 100); // 0 -> -1 -> 0 -> -1: a bounded 2-cycle
    }
}
