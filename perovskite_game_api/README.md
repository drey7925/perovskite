This crate contains the default game content for Perovskite, as well as APIs for extending that game content.

The API of this crate is unstable while Perovskite is in its early 0.x era. However, the API should be somewhat more
stable than the lower-level `perovskite_server` crate.

Most plugins should depend on this crate, which includes base content and various extension points.

If you're looking to *play* Perovskite, you can build the binary from this crate with default features enabled.