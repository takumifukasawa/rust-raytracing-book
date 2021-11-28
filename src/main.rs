// #![arrow(dead_code)]

extern crate image;
extern crate rayon;

mod rayt;

use crate::rayt::*;
use rayon::prelude::*;
// use std::{fs::File, io::prelude::*};
use image::{Rgb, RgbImage};

// struct Color ([f64; 3]);

const IMAGE_WIDTH: u32 = 200;
const IMAGE_HEIGHT: u32 = 100;

// fn save_ppm(filename: String, pixels: &[Color]) -> std::io::Result<()> {
//     let mut file = File::create(filename)?;
//     writeln!(file, "P3")?;
//     writeln!(file, "{} {}", IMAGE_WIDTH, IMAGE_HEIGHT)?;
//     writeln!(file, "255")?;
//     for Color([r, g, b]) in pixels {
//         // |x| ... はクロージャーの形式
//         let to255 = |x| (x * 255.99) as u8;
//         writeln!(file, "{} {} {}", to255(r), to255(g), to255(b))?;
//     }
//     file.flush()?;
//     Ok(())
// }

fn color(ray: &Ray) -> Color {
    let d = ray.direction.normalize();
    let t = 0.5 * (d.y() + 1.0);
    Color::new(0.5, 0.7, 1.0).lerp(Color::one(), t)
}

fn main() {
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
            let v = *y as f64 / (IMAGE_HEIGHT - 1) as f64;
            let ray = camera.ray(u, v);
            let rgb = color(&ray).to_rgb();
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        });
        img.save(String::from("render.png")).unwrap();
    // let mut pixels: Vec<Color> = Vec::with_capacity(IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize);
    // for j in 0..IMAGE_HEIGHT {
    //     let par_iter = (0..IMAGE_WIDTH).into_iter().map(|i| {
    //         Color([
    //             i as f64 / IMAGE_WIDTH as f64,
    //             j as f64 / IMAGE_HEIGHT as f64,
    //             0.5,
    //         ])
    //     });
    //     // underscore にすることでコンパイラの型推論が走ってる？
    //     let mut line_pixels: Vec<_> = par_iter.collect();
    //     // 型を明示的に指定する場合
    //     // let mut line_pixels: Vec<Color> = par_iter.collect();
    //     pixels.append(&mut line_pixels);
    // }
    // save_ppm(String::from("render.ppm"), &pixels).unwrap();
}
