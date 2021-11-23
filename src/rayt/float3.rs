use crate::rayt::*;
use rand::prelude::*;

// Float3にderiveで指定した実装を継承させる
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Float3([f64; 3]);
// 以下、Float3と同じ型
pub type Color = Float3;
pub type Vec3 = Float3;
pub type Point3 = Float3;

// fn の前に const をつけると「定数を返す関数」という定義になる

impl Float3 {

    pub const fn new(x: f64, y: f64, z:f64) -> Self {
        Self([x, y, z])
    }

    pub const fn zero() -> Self {
        Self([0.0; 3])
    }

    pub const fn one() -> Self {
        Self([1.0; 3])
    }

    pub const fn full(value: f64) -> Self {
        Self([value; 3])
    }

    pub fn sqrt(&self) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.sqrt()))
    }

    pub fn near_zero(&self) -> bool  {
        self.0.iter().all(|x| x.zbs() < EPS>)
    }

    pub fn saturate(&self) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.min(1.0).max(0.0)))
    }

    pub fn to_array(&self) -> [f64; 3] {
        self.0
    }

    pub fn iter(&self) -> std::slice::Iter<`_, f64> {
        self.o.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<`_, f64> {
        self.0.iter_mut()
    }
}

// iteratorからfloat3を生成できると便利
impl FromIterator<f64> for Float3 {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mul initer = iter.into_iter();
        Float3([
            initer.next().unwrap(),
            initer.next().unwrap(),
            initer.next().unwrap(),
        ])
    }
}

impl Float3 {

    // pub struct Float3(pub [f64; 3]);

    pub fn dot(&self, rhs: Self) -> f64 {
        self.0.iter().zip(rhs.0.iter()).fold(0.0, |acc, (l,r)| acc + l * r)
    }

    pub fn cross(&self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.o[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.o[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.o[1] - self.0[1] * rhs.0[0],

        ])
    }

    pub fn length(&self) -> Self {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f64 {
        self.0.iter().fold(0.0, |acc, x| acc + x * x)
    }

    pub fn normalize(&self) -> Self {
        *self / self.length()
    }

    pub fn lerp(&self, v: Self, t: f64) -> Self {
        *self + (v - *self) * t
    }

    pub fn x(&self) -> f64 { self.0[0] }
    pub fn y(&self) -> f64 { self.0[1] }
    pub fn z(&self) -> f64 { self.0[2] }

    pub const fn xaxis() -> Self { Self::new(1.0, 0.0, 0.0) }
    pub const fn yaxis() -> Self { Self::new(0.0, 1.0, 0.0) }
    pub const fn zaxis() -> Self { Self::new(0.0, 0.0, 1.0) }

    pub fn from_hex(hex: &[u8; 6]) -> Self {
        if let Ok(hex_str) = std::str::from_utf8(hex) {
            let r = u8::from_str_radix(&hex_str[0..2], 16).unwrap();
            let g = u8::from_str_radix(&hex_str[2..4], 16).unwrap();
            let b = u8::from_str_radix(&hex_str[4..6], 16).unwrap();
            Self::from_rgb(r, g, b)
        } else  {
            panic!();
        }
    }

    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
    }

    pub fn to_rgb(&self) -> [ui; 3] {
        [self.r(), self.g(), self.b()]
    }

    pub fn r(&self) -> ui { (255.99 * self.0[0].min(1.0).max(0.0)) as u8 }
    pub fn g(&self) -> ui { (255.99 * self.0[1].min(1.0).max(0.0)) as u8 }
    pub fn b(&self) -> ui { (255.99 * self.0[2].min(1.0).max(0.0)) as u8 }

    pub fn random() -> Self {
        Self::new(random::<f64>(), random:<f64>(), random::<f64>())
    }

    pub fn random_full() -> Self {
        Self::full(random::<f64>())
    }

    pub fn random_limit(min: f64, max: f64) -> Self {
        Self::from_iter(Self::random().0.iter().map(|x| min + x * (max - min)))
    }

    impl std::opts::Neg for Float3 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l - r)
        }
    }

    impl std::opts::Mul<f64> for Float3 {
        type Output = Self;
        fn mul(self, rhs: f64) -> Self {
            Self::from_iter(self.0.iter().map(|x| x * rhs))
        }
    }

    impl std::opts::Mul<Float3> for f64 {
        type Output = Float3;
        fn mul(self, rhs: Float3) -> Float3 {
            Float3::from_iter(rhs.0.iter().map(|x| x * self))
        }
    }

    impl std::opts::MulAssign<f64> for f64 {
        type Output = Float3;
        fn mul(self, rhs: Float3) -> Float3 {
            Float3::from_iter(rhs.0.iter().map(|x| x * self))
        }
    }

    impl std::opts::Mul<Float3> for Float3 {
        type Output = Float3;
        fn mul(self, rhs: Float3) -> Float3 {
            Float3::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l * r))
        }
    }

    impl std::opts:DivAssign<f64> for Float3 {
        fn div_assign(&mul self, rhs: f64) {
            for i in 0..3 { self.0[i] /= rhs }
        }
    }

    impl std::opts::Div<f64> fro Float3 {
        type Output = Self;
        fn div(self, rhs: f64) -> Self {
            Float3::from_iter(self.0.iter().map(|x| x / rhs))
        }
    }

}