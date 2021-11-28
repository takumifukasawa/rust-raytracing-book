use crate::rayt::*;

struct HitInfo {
    t: f64,
    p: Point3,
    n: Vec3,
}

impl HitInfo {
    const fn new(t: f64, p: Point3, n: Vec3) -> Self {
        Self { t, p, n }
    }
}

trait Shape: Sync {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo>;
}

struct Sphere  {
    center: Point3,
    radius: f64,
}

impl Sphere {
    const fn new (center: Point3, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Shape for Sphere {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
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
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * ray.direction.dot(oc);
        let c = oc.dot(oc) - self.radius.powi(2);
        let d = b * b - 4.0 * a * c;
        if d > 0.0 {
            let root = d.sqrt();
            // 判別式の(+-)の-の方が始点に近いので-を先に判定
            let temp = (-b - root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                return Some(HitInfo::new(temp, p, (p - self.center) / self.radius));
            }
            let temp = (-b + root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                return Some(HitInfo::new(temp, p, (p - self.center) / self.radius));
            }
        }

        None
    }
}

struct ShapeList {
    pub objects: Vec<Box<dyn Shape>>,
}

impl ShapeList {
    pub fn new () -> Self {
        Self { objects: Vec::new() }
    }
    pub fn push(&mut self, object: Box<dyn Shape>) {
        self.objects.push(object);
    }
}

impl Shape for ShapeList {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let mut hit_info: Option<HitInfo> = None;
        let mut closest_so_far = t1;
        for object in &self.objects {
            if let Some(info) = object.hit(ray, t0, closest_so_far) {
                closest_so_far = info.t;
                hit_info = Some(info);
            }
        }

        hit_info
    }
}

struct SimpleScene {
    world: ShapeList
}

impl SimpleScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        world.push(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));
        world.push(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));
        Self { world }
    }
    fn background(&self, d: Vec3) -> Color {
        let t = 0.5 * (d.y() + 1.0);
        Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
    }
}

impl Scene for SimpleScene {
    fn camera(&self) -> Camera {
        Camera::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-2.0, -1.0, -1.0),
        )
    }

    fn trace(&self, ray: Ray) -> Color {
        let hit_info = self.world.hit(&ray, 0.0, f64::MAX);
        if let Some(hit) = hit_info {
            0.5 * (hit.n + Vec3::one())
        } else {
            self.background(ray.direction)
        }
        // let c = Point3::new(0.0, 0.0, -1.0);
        // let t = self.hit_sphere(c, 0.5, &ray);
        // if t > 0.0 {
        //     // 法線は(ヒットした地点 - 中心位置)で求まる
        //     let n = (ray.at(t) - c).normalize();
        //     // [-1~1]->[0~1]にremap
        //     return 0.5 * (n + Vec3::one());
        // }
        // self.background(ray.direction)
    }
}
pub fn run() {
    render(SimpleScene::new());
}