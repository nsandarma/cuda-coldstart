#!/usr/bin/env python3
"""Run every cold-start configuration as a fresh process, many times.

Configurations are interleaved round-robin so thermal drift and background
noise spread evenly instead of biasing whichever config ran last.

Outputs (in results/):
  env.json      machine / driver / toolkit description
  raw.jsonl     one line per trial: config, stages, process wall time
  summary.csv   median / p10 / p90 per (config, stage)
  launch.jsonl  steady-state launch overhead runs
"""
import argparse
import csv
import glob
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(ROOT, "results")
CACHE_ROOT = os.path.join(RESULTS, "jit_cache")
KERNEL_COUNTS = [1, 32, 256]


def gpu_power_path():
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        try:
            vendor = open(f"{dev}/vendor").read().strip()
            cls = open(f"{dev}/class").read().strip()
        except OSError:
            continue
        if vendor == "0x10de" and cls.startswith("0x03"):
            return f"{dev}/power/runtime_status"
    return None


def read_power(path):
    try:
        return open(path).read().strip() if path else "unknown"
    except OSError:
        return "unknown"


def make_configs():
    cfgs = []
    for n in KERNEL_COUNTS:
        drv = os.path.join(BUILD, "bench_driver")
        src = os.path.join(BUILD, f"kernels_{n}.cu")
        cfgs += [
            dict(api="driver", variant="cubin", kernels=n, cache="none",
                 cmd=[drv, "cubin", os.path.join(BUILD, f"kernels_{n}.cubin")]),
            dict(api="driver", variant="ptx", kernels=n, cache="cold",
                 cmd=[drv, "ptx", os.path.join(BUILD, f"kernels_{n}.ptx")]),
            dict(api="driver", variant="ptx", kernels=n, cache="warm",
                 cmd=[drv, "ptx", os.path.join(BUILD, f"kernels_{n}.ptx")]),
            dict(api="driver", variant="nvrtc-ptx", kernels=n, cache="cold",
                 cmd=[drv, "nvrtc-ptx", src]),
            dict(api="driver", variant="nvrtc-cubin", kernels=n, cache="none",
                 cmd=[drv, "nvrtc-cubin", src]),
        ]
        for loading in ["LAZY", "EAGER"]:
            cfgs.append(dict(api="runtime", variant="sass", kernels=n,
                             loading=loading, cache="none",
                             cmd=[os.path.join(BUILD, f"rt_sass_{n}")]))
            for cache in ["cold", "warm"]:
                cfgs.append(dict(api="runtime", variant="ptx", kernels=n,
                                 loading=loading, cache=cache,
                                 cmd=[os.path.join(BUILD, f"rt_ptx_{n}")]))
    for c in cfgs:
        c["name"] = "/".join(
            str(x) for x in [c["api"], c["variant"], f"k{c['kernels']}",
                             c.get("loading", "-"), f"cache-{c['cache']}"])
    return cfgs


def env_for(cfg):
    env = dict(os.environ)
    env.pop("CUDA_CACHE_DISABLE", None)
    env.pop("CUDA_CACHE_PATH", None)
    if "loading" in cfg:
        env["CUDA_MODULE_LOADING"] = cfg["loading"]
    if cfg["cache"] == "warm":
        env["CUDA_CACHE_PATH"] = os.path.join(CACHE_ROOT,
                                              cfg["name"].replace("/", "_"))
    else:
        # cubin/SASS never touch the JIT cache; disabling it keeps "cold"
        # cold and makes "none" immune to leftovers from other runs.
        env["CUDA_CACHE_DISABLE"] = "1"
    return env


def run_once(cfg, power_path):
    power = read_power(power_path)
    start = time.perf_counter()
    proc = subprocess.run(cfg["cmd"], env=env_for(cfg), capture_output=True,
                          text=True)
    wall_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg['name']} failed:\n{proc.stderr}")
    stages = {}
    for line in proc.stdout.splitlines():
        rec = json.loads(line)
        stages[rec["stage"]] = rec["ms"]
    return dict(power_before=power, wall_ms=wall_ms, stages=stages)


def collect_env():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            return ""

    return dict(
        gpu=sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,"
               "memory.total --format=csv,noheader"),
        cuda_toolkit=sh("nvcc --version | tail -1"),
        cpu=sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
        kernel=platform.release(),
        other_gpu_processes=sh("nvidia-smi --query-compute-apps=pid,name "
                               "--format=csv,noheader"),
        runtime_d3=sh("grep 'Runtime D3' /proc/driver/nvidia/gpus/*/power"),
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(trials, cfgs):
    rows = []
    for cfg in cfgs:
        runs = [t for t in trials if t["name"] == cfg["name"]]
        stage_names = list(runs[0]["stages"].keys()) + ["sum_of_stages",
                                                        "process_wall"]
        for stage in stage_names:
            if stage == "sum_of_stages":
                xs = [sum(r["stages"].values()) for r in runs]
            elif stage == "process_wall":
                xs = [r["wall_ms"] for r in runs]
            else:
                xs = [r["stages"][stage] for r in runs]
            rows.append(dict(
                name=cfg["name"], api=cfg["api"], variant=cfg["variant"],
                kernels=cfg["kernels"], loading=cfg.get("loading", ""),
                cache=cfg["cache"], stage=stage, n=len(xs),
                median_ms=round(statistics.median(xs), 4),
                p10_ms=round(percentile(xs, 0.1), 4),
                p90_ms=round(percentile(xs, 0.9), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="idle seconds before each trial (lets the GPU "
                         "power down, to measure wake-up cost)")
    ap.add_argument("--tag", default="default",
                    help="suffix for output files, e.g. 'idle-gpu'")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    power_path = gpu_power_path()
    cfgs = make_configs()

    with open(os.path.join(RESULTS, f"env_{args.tag}.json"), "w") as f:
        json.dump(collect_env(), f, indent=2)

    # Warm-up round (discarded): fills the OS page cache for libcuda/libnvrtc
    # and populates the JIT cache for the "warm" configs.
    print(f"warm-up: {len(cfgs)} configs")
    for cfg in cfgs:
        run_once(cfg, power_path)

    trials = []
    for rnd in range(args.trials):
        order = cfgs[:]
        random.shuffle(order)
        for cfg in order:
            if args.gap:
                time.sleep(args.gap)
            res = run_once(cfg, power_path)
            res.update(name=cfg["name"], round=rnd)
            trials.append(res)
        print(f"round {rnd + 1}/{args.trials} done")

    with open(os.path.join(RESULTS, f"raw_{args.tag}.jsonl"), "w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    rows = summarize(trials, cfgs)
    with open(os.path.join(RESULTS, f"summary_{args.tag}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(RESULTS, f"launch_{args.tag}.jsonl"), "w") as f:
        for _ in range(5):
            out = subprocess.run([os.path.join(BUILD, "bench_launch")],
                                 capture_output=True, text=True, check=True)
            f.write(out.stdout)

    print("\nsum_of_stages (median ms):")
    for r in rows:
        if r["stage"] == "sum_of_stages":
            print(f"  {r['name']:<45} {r['median_ms']:>10.1f}")


if __name__ == "__main__":
    main()
