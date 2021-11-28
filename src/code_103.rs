extern crate image;
extern crate rayon;

use crate::rayt::*;
use image::{Rgb, RgbImage};
use rayon::prelude::*;

const IMAGE_WIDTH: u32 = 200;
const IMAGE_HEIGHT: u32 = 100;

fn hit_sphere(center: Point3, radius: f64, ray: &Ray) -> bool {
    /*
     * x^2 + y^2 + z^2 = r^2;
     * 
     * // 中心位置をcx,cy,czとすると
     * (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2
     * 
     * // 位置をx,y,zとすると
     * (px - cx)^2 + (pz - cy)^2 + (pz - cz)^2 = r^2
     * a: (p - c)^2 = r^2
     * 
     * // rayの方程式
     * p(t) = o + td;
     * 
     * // a に代入
     * (p(t) - c)^2 = r^2
     * (o + td - c)^2 = r^2
     * 
     * oc = o - cとすると
     * (oc + td)^2 = r^2
     * (oc + td)^2 - r^2 = 0
     * 
     * tについて展開
     * (d*d)t^2 + 2(d + oc)t + (oc)^2 - r^2 = 0
     * 
     * ax^2 + bx + c = 0 の二次方程式に判別式Dを当てはめると、
     * 
     * D = b^2 - 4ac
     * a = (d * d)
     * b = 2(d + oc)
     * c = (oc)^2 - r^2 = (o - c)^2 - r^2
     * 
     */
    // a = d dot d
    // b = 2 * (d dot (o - c))
    // c = ((o - c) dot (o - c)) - r ^ 2
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * ray.direction.dot(oc);
    let c = oc.dot(oc) - radius.powi(2);
    // 二次方程式の解の判別式
    let d = b * b - 4.0 * a * c;
    d > 0.0
}

fn color(ray: &Ray) -> Color {
    if hit_sphere(Point3::new(0.0, 0.0, -1.0), 0.5, &ray) {
        return Color::new(1.0, 0.0, 0.0);
    }
    let d = ray.direction.normalize();
    let t = 0.5 * (d.y() + 1.0);
    Color::new(0.5, 0.7, 1.0).lerp(Color::one(), t)
}

pub fn run() {
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
}
