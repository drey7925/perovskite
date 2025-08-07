fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("src/icon_tentative_0.2.0.ico");
        res.compile().unwrap();
    }
}
