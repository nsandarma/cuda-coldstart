#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using Clock = std::chrono::steady_clock;

// Measures consecutive stages: each lap() returns ms since the previous lap.
struct StageTimer {
  Clock::time_point last = Clock::now();

  double lap() {
    Clock::time_point now = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    return ms;
  }
};

// One JSON object per line on stdout; run_bench.py parses these.
inline void emit(const char *stage, double ms) {
  printf("{\"stage\":\"%s\",\"ms\":%.4f}\n", stage, ms);
  fflush(stdout);
}

inline std::string read_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(EXIT_FAILURE);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

constexpr int kN = 1 << 20;
constexpr int kBlock = 256;
