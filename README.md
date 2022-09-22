Look into performance analyzers
https://crates.io/crates/criterion

create branch and try this using same criterion crate
try using ndarray instad of current library to compare speed (https://docs.rs/ndarray/0.12.1/ndarray/)

implement ReLu
    mninst data label wouldnt only need to be one node with label value
    because ReLu is linear greater than x, so could guess numerical values