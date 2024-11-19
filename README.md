# Rust Unscented Kalman Filter

This is a small test project to test Rust's capabilities in signal processing and generic programming. The goal is to build a matrix based signal filter of generic size. The programm makes heavy use of Rust's typesystem to check at compiletime if the matrix dimensions are correct. 
The Unscented Kalman Filter can be used to observe non-linear time-invariant systems. The unittest simulates a simple non-linear system and writes the simulated and the observed system state into a CSV file.

## Run from Source

```bash
git clone https://github.com/paul-roettger/r-signal-fold.git
cd r-signal-fold
cargo test
```