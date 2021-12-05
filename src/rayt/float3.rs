use crate::rayt::*;
use rand::prelude::*;
use std::iter::FromIterator;

// Float3にderiveで指定した実装を継承させる
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Float3([f64; 3]);

// 以下、Float3と同じ型
pub type Color = Float3;
pub type Vec3 = Float3;
pub type Point3 = Float3;

// fn の前に const をつけると「定数を返す関数」という定義になる

impl Float3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
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

    pub fn near_zero(&self) -> bool {
        self.0.iter().all(|x| x.abs() < EPS)
    }

    pub fn saturate(&self) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.min(1.0).max(0.0)))
    }

    pub fn to_array(&self) -> [f64; 3] {
        self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.0.iter_mut()
    }
}

/*
 * impl
 * メソッド定義。関数と同じくfnを使って定義
 */

// iteratorからfloat3を生成できると便利
impl FromIterator<f64> for Float3 {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut initer = iter.into_iter();
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
        self.0
            .iter()
            .zip(rhs.0.iter())
            .fold(0.0, |acc, (l, r)| acc + l * r)
    }

    pub fn cross(&self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }

    pub fn length(&self) -> f64 {
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

    pub fn x(&self) -> f64 {
        self.0[0]
    }
    pub fn y(&self) -> f64 {
        self.0[1]
    }
    pub fn z(&self) -> f64 {
        self.0[2]
    }

    pub const fn xaxis() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }
    pub const fn yaxis() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }
    pub const fn zaxis() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn from_hex(hex: &[u8; 6]) -> Self {
        if let Ok(hex_str) = std::str::from_utf8(hex) {
            let r = u8::from_str_radix(&hex_str[0..2], 16).unwrap();
            let g = u8::from_str_radix(&hex_str[2..4], 16).unwrap();
            let b = u8::from_str_radix(&hex_str[4..6], 16).unwrap();
            Self::from_rgb(r, g, b)
        } else {
            panic!();
        }
    }

    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
    }

    pub fn to_rgb(&self) -> [u8; 3] {
        [self.r(), self.g(), self.b()]
    }

    pub fn r(&self) -> u8 {
        (255.99 * self.0[0].min(1.0).max(0.0)) as u8
    }
    pub fn g(&self) -> u8 {
        (255.99 * self.0[1].min(1.0).max(0.0)) as u8
    }
    pub fn b(&self) -> u8 {
        (255.99 * self.0[2].min(1.0).max(0.0)) as u8
    }

    pub fn random() -> Self {
        Self::new(random::<f64>(), random::<f64>(), random::<f64>())
    }

    pub fn random_full() -> Self {
        Self::full(random::<f64>())
    }

    pub fn random_limit(min: f64, max: f64) -> Self {
        Self::from_iter(Self::random().0.iter().map(|x| min + x * (max - min)))
    }

    pub fn random_in_unit_sphere() -> Self {
        /*
         * 長さが1以下になるまでloop
         * => 条件を満たすまで一様関数を何度も生成して繰り返すこと = 棄却法
         */
        loop {
            let point = Self::random_limit(-1.0, 1.0);
            if point.length_squared() < 1.0 {
                return point;
            }
        }
    }

    // linear -> sRGB
    pub fn gamma(&self, factor: f64) -> Self {
        let recip = factor.recip();
        Self::from_iter(self.0.iter().map(|x| x.powf(recip)))
    }

    // sRGB -> linear
    pub fn degamma(&self, factor: f64) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.powf(factor)))
    }

    /*
     *        \    |n   /|       
     *         \   |  r/ |        
     *        v \  |  /  |b       
     *           \ | /   |        
     *            \|/    |        
     * --------------------------
     *              \    |       
     *               \   |       
     *               v\  |b      
     *                 \ |       
     *                  \| 
     * 
     * rが反射ベクトル
     * 
     * r = v + 2b
     * b = -(v dot n) * n
     * 
     * r = v + -2 * (v dot n) * n
     */
    pub fn reflect(&self, normal: Self) -> Self {
        *self - 2.0 * self.dot(normal) * normal
    }

    /**
     * ref:
     * https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5
     *
     * スネルの法則 
     * n1sinΘ1 = n2sinΘ2
     */
    pub fn refract(&self, normal: Self, ni_over_nt: f64) -> Option<Float3> {
        let uv = self.normalize();
        let dt = uv.dot(normal);
        /* d: 判別式
         * スネルの法則を置き換える
         * sinΘ2 = (n1 / n2)sinΘ1
         * Θ1 = 0~90 なので sinΘ1 = 0~90
         * n1 < n2 ... sinΘ2 = 0~1
         * n1 > n2 ... (n1 / n2) > 1 なので sinΘ1 が大きい場合に sinΘ2 > 1 になってしまう
         *  => 屈折光がなくなり反射光だけになる = 全反射
         * 屈折をする媒質を誘導体と呼ぶ
         */
        let d = 1.0 - ni_over_nt.powi(2) * (1.0 - dt.powi(2));
        if d > 0.0 {
            Some(-ni_over_nt * (uv - normal * dt) - normal * d.sqrt())
        } else {
            None
        }
    }

    // 半球上の無作為な方向
    pub fn random_cosine_direction() -> Self {
        let Self([r1, r2, _]) = Self::random();
        let z = (1.0 - r2).sqrt();
        let (x, y) = (PI2 * r1).sin_cos();
        let r2sqrt = r2.sqrt();
        Self::new(x * r2sqrt, y * r2sqrt, z)
    }

    pub fn random_to_sphere(radius: f64, distance_squared: f64) -> Self {
        let Self([rx, ry, _]) = Self::random();
        let rr = radius.powi(2).min(distance_squared);
        let cos_theta_max = (1.0 - rr * distance_squared.recip()).sqrt();
        let z = 1.0 - ry * (1.0 - cos_theta_max);
        let sqrtz = (1.0 - z.powi(2)).sqrt();
        let (x, y) = (PI2 * rx).sin_cos();
        Self::new(x * sqrtz, y * sqrtz, z)
    }
}

impl std::ops::Neg for Float3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_iter(self.iter().map(|x| -x))
    }
}

impl std::ops::AddAssign<Float3> for Float3 {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..3 {
            self.0[i] += rhs.0[i]
        }
    }
}

impl std::ops::Add<Float3> for Float3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l + r))
    }
}

impl std::ops::SubAssign<Float3> for Float3 {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..3 {
            self.0[i] -= rhs.0[i]
        }
    }
}

impl std::ops::Sub<Float3> for Float3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l - r))
    }
}

impl std::ops::Mul<f64> for Float3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::from_iter(self.0.iter().map(|x| x * rhs))
    }
}

impl std::ops::Mul<Float3> for f64 {
    type Output = Float3;
    fn mul(self, rhs: Float3) -> Float3 {
        Float3::from_iter(rhs.0.iter().map(|x| x * self))
    }
}

impl std::ops::MulAssign<f64> for Float3 {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..3 {
            self.0[i] *= rhs
        }
    }
}

impl std::ops::Mul<Float3> for Float3 {
    type Output = Float3;
    fn mul(self, rhs: Float3) -> Float3 {
        Float3::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l * r))
    }
}

impl std::ops::DivAssign<f64> for Float3 {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..3 {
            self.0[i] /= rhs
        }
    }
}

impl std::ops::Div<f64> for Float3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Float3::from_iter(self.0.iter().map(|x| x / rhs))
    }
}