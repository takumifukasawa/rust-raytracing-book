/*
 * use crate::rayt::* でアクセスできるようになるための実装?
 * rustにおける folder/mod.rs は、
 * jsで言うところの folder/index.js で諸々のmoduleをimport/exportしているファイルという認識で合ってる?
 */

// import raytracing modules
mod camera;
mod float3;
mod quat;
mod ray;

// export modules
pub use self::float3::{Color, Float3, Point3, Vec3};

pub use self::camera::Camera;
pub use self::quat::Quat;
pub use self::ray::Ray;

pub use std::f64::consts::FRAC_1_PI;
pub use std::f64::consts::PI;
pub const PI2: f64 = PI * 2.0;
pub const EPS: f64 = 1e-6;
