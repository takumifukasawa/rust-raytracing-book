use crate::rayt::*;

struct HitInfo {
    t: f64,
    p: Point3,
    n: Vec3,
    m: Arc<dyn Material>,
}

impl HitInfo {
    fn new(t: f64, p: Point3, n: Vec3, m: Arc<dyn Material>) -> Self {
        Self { t, p, n, m }
    }
}

struct ScatterInfo {
    ray: Ray, // 散乱後の新しい光線
    albedo: Color, // 反射率
}

impl ScatterInfo {
    fn new(ray: Ray, albedo: Color) -> Self {
        Self { ray, albedo }
    }
}

// Send ... 所有権を移動できることの明示
trait Material: Sync + Send {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo>;
}

struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

struct Metal {
    albedo: Color,
    fuzz: f64
}

impl Metal {
    fn new(albedo: Color, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let target = hit.p + hit.n + Vec3::random_in_unit_sphere();
        Some(ScatterInfo::new(Ray::new(hit.p, target - hit.p), self.albedo))
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let mut reflected = ray.direction.normalize().reflect(hit.n);
        reflected = reflected + self.fuzz * Vec3::random_in_unit_sphere();
        // 反射ベクトルと法線の角度が0より大きいときにscatterInfoを返す
        if reflected.dot(hit.n) > 0.0 {
            Some(ScatterInfo::new(Ray::new(hit.p, reflected), self.albedo))
        } else {
            None
        }
    }
}

// Sync ... 複数スレッドから参照されても安全なことを意味する
trait Shape: Sync {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo>;
}

struct Sphere {
    center: Point3,
    radius: f64,
    material: Arc<dyn Material>
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Arc<dyn Material>) -> Self {
        Self { center, radius, material }
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
                return Some(HitInfo::new(
                    temp,
                    p,
                    (p - self.center) / self.radius,
                    Arc::clone(&self.material)
                ));
            }
            let temp = (-b + root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                return Some(HitInfo::new(
                    temp,
                    p,
                    (p - self.center) / self.radius,
                    Arc::clone(&self.material)
                ));
            }
        }

        None
    }
}

struct ShapeList {
    /*
     * Box<T>はヒープメモリに格納する
     * dyn Trait は、トレイトオブジェクトであることを明示するための記法
     */
    pub objects: Vec<Box<dyn Shape>>,
}

impl ShapeList {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
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
    world: ShapeList,
}

impl SimpleScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        world.push(Box::new(Sphere::new(
            Point3::new(0.6, 0.0, -1.0),
            0.5,
            Arc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)))
        )));
        world.push(Box::new(Sphere::new(
            Point3::new(-0.6, 0.0, -1.0),
            0.5,
            Arc::new(Metal::new(Color::new(0.8, 0.8, 0.8), 1.0))
        )));
        world.push(Box::new(Sphere::new(
            Point3::new(0.0, -100.5, -1.0),
            100.0,
            Arc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)))
        )));
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
        let hit_info = self.world.hit(&ray, 0.0001, f64::MAX);
        if let Some(hit) = hit_info {
            // let target = hit.p + hit.n + Vec3::random_in_unit_sphere();
            // 0.5 * self.trace(Ray::new(hit.p, target - hit.p))
            let scatter_info = hit.m.scatter(&ray, &hit);
            if let Some(scatter) = scatter_info {
                return scatter.albedo * self.trace(scatter.ray);
            } else {
                return Color::zero();
            }
        } else {
            self.background(ray.direction)
        }
    }
}
pub fn run() {
    // render(SimpleScene::new());
    render_aa(SimpleScene::new());
}
