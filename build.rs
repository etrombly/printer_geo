use shaderc::{CompileOptions, Compiler, ShaderKind};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("IO error")]
    IO(#[from] std::io::Error),
    #[error("Shaderc error")]
    Shaderc(#[from] shaderc::Error),
    #[error("No filename for glsl")]
    NoFile,
    #[error("Couldn't find shaderc compiler")]
    NoCompiler,
    #[error("Couldn't create shaderc options")]
    NoOptions,
}

fn main() -> Result<(), BuildError> {
    // Tell the build script to only run again if we change our source shaders
    println!("cargo:rerun-if-changed=shaders/glsl");
    std::fs::create_dir_all("shaders/spirv")?;

    for entry in std::fs::read_dir("shaders/glsl")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let in_path = entry.path();

            let shader_type = in_path
                .extension()
                .and_then(|ext| match ext.to_string_lossy().as_ref() {
                    "vert" => Some(ShaderKind::Vertex),
                    "frag" => Some(ShaderKind::Fragment),
                    "comp" => Some(ShaderKind::Compute),
                    _ => None,
                });
            if let Some(shader_type) = shader_type {
                let source = std::fs::read_to_string(&in_path)?;
                let mut compiler = Compiler::new().ok_or(BuildError::NoCompiler)?;
                let mut options = CompileOptions::new().ok_or(BuildError::NoOptions)?;
                options.add_macro_definition("EP", Some("main"));
                let binary_result = compiler.compile_into_spirv(
                    &source,
                    shader_type,
                    &in_path.to_string_lossy(),
                    "main",
                    Some(&options),
                )?;

                // Determine the output path based on the input name
                let out_path = format!(
                    "shaders/spirv/{}.spv",
                    entry.path().file_stem().ok_or(BuildError::NoFile)?.to_string_lossy()
                );

                std::fs::write(&out_path, &binary_result.as_binary_u8())?;
            }
        }
    }
    Ok(())
}
