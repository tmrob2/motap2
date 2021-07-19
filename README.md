## Installation
Requirements
- Valid Rust Installation - Toolchain management with Rustup
- Gurobi 9.1 Installation - Gurobi can be installed with an Academic License

## Usage

To solve a MOTAP problem a DFA and MDP must be defined at the application binary level. 
There is an example in ```motap.rs``` which includes a ```send task``` DFA, and a ```send agent``` MDP.
This gives an idea how the agent MDP, and DFA should be constructed.

The STAPU framework follows a similar design. Examples for STAPU DFA models, and MDPs are
given in ```stapu.rs```.

The output scheduler graph will be inserted into ```../diagnostics/merged_scheduler.dot```
and can be converted to a ```pdf``` graph with ```Graphviz``` in the usual way.

To run a binary: build -> run with Cargo.

```
cargo build --bin motap --release
cargo run --bin motap --release
```

