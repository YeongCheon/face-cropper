use std::{fs, path::Path, time::Duration};

use clap::{Arg, Command};
use image::imageops::crop_imm;
use indicatif::ProgressBar;
use rust_faces::{
    FaceDetection, FaceDetector, FaceDetectorBuilder, InferParams, MtCnnParams, Provider, ToArray3,
    ToRgb8,
};

fn get_file_list(input_path: &String) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let valid_extensions = vec!["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "avif"];

    let path = Path::new(input_path);
    if !path.exists() {
        eprintln!("invalid path: {}", input_path);
        std::process::exit(1);
    }

    if path.is_file() {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            let ext_lower = ext.to_lowercase();

            if valid_extensions.contains(&ext_lower.as_str()) {
                res.push(path.display().to_string());
            } else {
                eprintln!("입력된 파일은 이미지 파일이 아닙니다: {}", path.display());
                std::process::exit(1);
            }
        } else {
            eprintln!("파일의 확장자를 확인할 수 없습니다: {}", path.display());
            std::process::exit(1);
        }
    } else if path.is_dir() {
        match fs::read_dir(path) {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let item = entry.path().display().to_string();
                            res.push(item);
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
            }
            Err(err) => eprintln!("Error: {}", err),
        }
    } else {
        eprintln!("유효한 파일 또는 디렉토리가 아닙니다: {}", input_path);
        std::process::exit(1);
    }

    res
}

fn crop_faces(face_detector: &Box<dyn FaceDetector>, path_list: Vec<String>) {
    path_list.iter().for_each(|image_path| {
        let bar = ProgressBar::new_spinner().with_message(format!("{}", image_path));
        bar.enable_steady_tick(Duration::from_micros(100));

        let path = Path::new(image_path);
        let file_name = path.file_name().unwrap();
        let extension = match path.extension() {
            Some(extension) => extension.to_str().unwrap(),
            None => "jpg",
        };

        let image = match image::open(path) {
            Ok(image) => image.into_rgb8().into_array3(),
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };

        let faces = face_detector.detect(image.view().into_dyn()).unwrap();

        let output_dir = "output";
        std::fs::create_dir_all(output_dir).expect("Can't create test output dir.");
        faces.iter().enumerate().for_each(|(idx, face)| {
            let image = image.to_rgb8();

            let rect = face.rect;

            let output_image = crop_imm(
                &image,
                rect.x as u32,
                rect.y as u32,
                rect.width as u32,
                rect.height as u32,
            )
            .to_image();

            let output_name = format!(
                "{}.{}",
                format!("{}_output{:0>3}", file_name.to_str().unwrap(), idx + 1),
                extension
            );

            output_image
                .save(format!("{}/{}", output_dir, output_name))
                .expect("Can't save test image.");
        });

        bar.finish();
    });
}

fn main() {
    let matches = Command::new("Face Cropper")
        .version("0.1")
        .author("YeongCheon Kim")
        .about("Face crop from your image")
        .arg(Arg::new("path").help("path").index(1))
        .get_matches();

    let input_path = matches
        .get_one::<String>("path")
        .expect("경로가 제공되지 않습니다.");

    let face_detector = FaceDetectorBuilder::new(FaceDetection::MtCnn(MtCnnParams::default()))
        .download()
        .infer_params(InferParams {
            provider: Provider::OrtCoreMl,
            intra_threads: Some(5),
            ..Default::default()
        })
        .build()
        .expect("Fail to load the face detector.");

    let path_list = get_file_list(input_path);
    crop_faces(&face_detector, path_list);
}
