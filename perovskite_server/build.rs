use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR");
    let out_path = PathBuf::from(out_dir).join("build_info.txt");
    let mut out_file = BufWriter::new(File::create(&out_path).expect("Create file"));
    for env in [
        "CARGO_PKG_VERSION",
        "TARGET",
        "HOST",
        "NUM_JOBS",
        "OPT_LEVEL",
        "PROFILE",
        "RUSTC_LINKER",
    ] {
        let val = env::var(env).unwrap_or(String::from("<NONE>"));
        out_file
            .write_fmt(format_args!("{}={}\n", env, val))
            .expect("write to file");
    }
    match rustc_version::version_meta() {
        Ok(v) => {
            out_file
                .write_fmt(format_args!(
                    "rustc: {} {} ({:?}) @ {} for {}\n llvm {}",
                    v.short_version_string,
                    v.semver,
                    v.channel,
                    v.commit_hash.as_deref().unwrap_or("???"),
                    v.host,
                    v.llvm_version
                        .map(|x| x.to_string())
                        .unwrap_or("???".to_string()),
                ))
                .expect("write to file");
        }
        Err(e) => {
            out_file
                .write_fmt(format_args!("rustc version unknown: {:?}", e))
                .expect("write to file");
        }
    }
}
