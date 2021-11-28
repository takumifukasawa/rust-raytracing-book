extern crate image;
extern crate rayon;

use crate::rayt::*;
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::{fs, path::Path};

const OUTPUT_FILENAME: &str = "render.png";
const BACKUP_FILENAME: &str = "render_bak.png";

const IMAGE_WIDTH: u32 = 200;
const IMAGE_HEIGHT: u32 = 100;

const SAMPLES_PER_PIXEL: usize = 8;
const GAMMA_FACTOR: f64 = 2.2;

fn backup() {
    let output_path = Path::new(OUTPUT_FILENAME);
    if output_path.exists() {
        println!("backup {:?} -> {:?}", OUTPUT_FILENAME, BACKUP_FILENAME);
        fs::rename(OUTPUT_FILENAME, BACKUP_FILENAME).unwrap();
    }
}

pub trait Scene {
    fn camera(&self) -> Camera;
    fn trace(&self, ray: Ray) -> Color;
    fn width(&self) -> u32 {
        IMAGE_WIDTH
    }
    fn height(&self) -> u32 {
        IMAGE_HEIGHT
    }
    fn spp(&self) -> usize {
        SAMPLES_PER_PIXEL
    }
    fn aspect(&self) -> f64 {
        self.width() as f64 / self.height() as f64
    }
}

pub fn render_aa(scene: impl Scene + Sync) {
    // window関連エラーになるので一旦使わない
    // backup()

    let camera = Camera::new(
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-2.0, -1.0, -1.0),
    );
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);
    img.enumerate_pixels_mut()
        .collect::<Vec<(u32, u32, &mut Rgb<u8>)>>()
        .par_iter_mut()
        .for_each(|(x, y, pixel)| {
            let mut pixel_color = (0..scene.spp()).into_iter().fold(Color::zero(), |acc, _| {
                let [rx, ry, _] = Float3::random().to_array();
                let u = (*x as f64 + rx) / (scene.width() - 1) as f64;
                // x,yは左上から。座標系は右上に上がっていくのでyを反転
                let v = ((scene.height() - *y - 1) as f64 + ry) / (scene.height() - 1) as f64;
                let ray = camera.ray(u, v);
                acc + scene.trace(ray)
            });
            pixel_color /= scene.spp() as f64;
            // let rgb = pixel_color.to_rgb();
            let rgb = pixel_color.gamma(GAMMA_FACTOR).to_rgb();
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        });
    img.save(String::from(OUTPUT_FILENAME)).unwrap();

    // なぜかエラー出る
    // draw_in_window(BACKUP_FILENAME, img).unwrap();
}

pub fn render(scene: impl Scene + Sync) {
    // window関連エラーになるので一旦使わない
    // backup()

    let camera = Camera::new(
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-2.0, -1.0, -1.0),
    );
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);
    img.enumerate_pixels_mut()
        .collect::<Vec<(u32, u32, &mut Rgb<u8>)>>()
        .par_iter_mut()
        .for_each(|(x, y, pixel)| {
            let u = *x as f64 / (IMAGE_WIDTH - 1) as f64;
            // x,yは左上から。座標系は右上に上がっていくのでyを反転
            let v = (IMAGE_HEIGHT - *y - 1) as f64 / (IMAGE_HEIGHT - 1) as f64;
            let ray = camera.ray(u, v);
            let rgb = scene.trace(ray).to_rgb();
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        });
    img.save(String::from(OUTPUT_FILENAME)).unwrap();

    // なぜかエラー出る
    // draw_in_window(BACKUP_FILENAME, img).unwrap();
}
