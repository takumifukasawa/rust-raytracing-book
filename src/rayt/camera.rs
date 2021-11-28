use crate::rayt::*;

#[derive(Debug)]
pub struct Camera {
    pub origin: Point3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

impl Camera {
    // 右手座標系
    // 横、縦、奥行
    pub fn new(u: Vec3, v: Vec3, w: Vec3) -> Self {
        Self {
            origin: Point3::zero(),
            u,
            v,
            w,
        }
    }

    pub fn from_lookat(origin: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64) -> Self {
        /*
         * 
         * bu,bv,bw は基底ベクトル
         * 横幅を2w,縦幅を2hとすると、中心から横幅の大きさはw,縦幅はh
         * bu = 2wx
         * bv = 2hy
         * 
         * 基底ベクトルからスクリーン上の位置pは
         * p = bu * u + bv * v + bw
         *
         * bwについて解くと、
         * bw = p - bu * u - bv * v;
         * 
         * oはカメラの位置
         * p = o - z;
         *
         * bw = o - z - bu * u - bv * v;
         * bw = o - bu * u - bv * v - z; 
         * 
         * bw = o - w * x - h * y - z; <- ここの変換がまだよくわかってない
         */
        let halfh = (vfov.to_radians() * 0.5).tan();
        let halfw = aspect * halfh;
        let w = (origin - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        let uw = halfw * u;
        let vh = halfh * v;
        Self {
            origin,
            u: 2.0 * uw,
            v: 2.0 * vh,
            w: origin - uw - vh - w,
        }
    }

    pub fn ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.w + self.u * u + self.v * v - self.origin,
        }
    }
}
