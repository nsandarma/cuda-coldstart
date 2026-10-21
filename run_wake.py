#!/usr/bin/env python3
"""Measure how the GPU's power state changes CUDA init cost on a laptop.

States, all with the same tiny program (1 kernel, precompiled SASS):
  suspended       GPU powered off by runtime PM (D3cold), nothing using it
  active          GPU powered on, but no other CUDA process
  other_client    another process holds a CUDA context (hold_gpu)

Output: results/wake_<tag>.jsonl and a printed median table.
"""
import argparse
import json
import os
import statistics
import subprocess
import time

from run_bench import BUILD, RESULTS, gpu_power_path, read_power

PROGRAMS = {
    "driver": [os.path.join(BUILD, "bench_driver"), "cubin",
               os.path.join(BUILD, "kernels_1.cubin")],
    "runtime": [os.path.join(BUILD, "rt_sass_1")],
}


def wait_for(power_path, wanted, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if read_power(power_path) == wanted:
            return True
        time.sleep(0.25)
    return False


def run(cmd):
    env = dict(os.environ, CUDA_CACHE_DISABLE="1")
    start = time.perf_counter()
    out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         check=True).stdout
    wall_ms = (time.perf_counter() - start) * 1000
    stages = {}
    for line in out.splitlines():
        rec = json.loads(line)
        stages[rec["stage"]] = rec["ms"]
    return wall_ms, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    power_path = gpu_power_path()
    if not power_path:
        raise SystemExit("no NVIDIA GPU runtime PM status found")
    records = []

    def record(state, program, trial):
        before = read_power(power_path)
        wall_ms, stages = run(PROGRAMS[program])
        rec = dict(state=state, program=program, trial=trial,
                   power_before=before, wall_ms=wall_ms, stages=stages)
        records.append(rec)
        print(json.dumps(rec), flush=True)

    for trial in range(args.trials):
        for program in PROGRAMS:
            if not wait_for(power_path, "suspended"):
                print("warning: GPU never suspended; skipping", flush=True)
            else:
                record("suspended", program, trial)

            # The GPU is now awake from the run above, with no clients left.
            record("active", program, trial)

            holder = subprocess.Popen([os.path.join(BUILD, "hold_gpu")],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            assert holder.stdout.readline().strip() == "ready"
            record("other_client", program, trial)
            holder.stdin.close()
            holder.wait()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"wake_{args.tag}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("\nmedian ms (first stage = what the user waits on):")
    for program in PROGRAMS:
        for state in ["suspended", "active", "other_client"]:
            rs = [r for r in records
                  if r["program"] == program and r["state"] == state]
            if not rs:
                continue
            first = list(rs[0]["stages"].keys())[0]
            init = statistics.median(r["stages"][first] for r in rs)
            total = statistics.median(sum(r["stages"].values()) for r in rs)
            print(f"  {program:<8} {state:<13} {first}={init:8.1f}  "
                  f"all_stages={total:8.1f}  (n={len(rs)})")


if __name__ == "__main__":
    main()
