fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .compile_protos(
            &[
                "proto/auth.proto",
                "proto/bundle.proto",
                "proto/searcher.proto",
                "proto/packet.proto",
                "proto/shared.proto",
                "proto/yellowstone.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
