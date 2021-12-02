
use crate::rayt::*;

trait Texture: Sync + Send {
    fn value(&self, u: f64, v: f64, p: Point3) -> Color;
}

struct ColorTexture {
    color: Color
}

struct CheckerTexture {
    odd: Box<dyn Texture>,
    even: Box<dyn Texture>,
    freq: f64
}

impl CheckerTexture {
    fn new(odd: Box<dyn Texture>, even: Box<dyn Texture>, freq: f64) -> Self {
        Self { odd, even, freq }
    }
}

impl ColorTexture {
    const fn new(color: Color) -> Self {
        Self { color }
    }
}

impl Texture for ColorTexture {
    fn value(&self, _u: f64, _v: f64, _p: Point3) -> Color {
        self.color
    }
}

impl Texture for CheckerTexture {
    fn value(&self, u: f64, v: f64, p: Point3) -> Color {
        let sines = p.iter().fold(1.0, |acc, x| acc * (x * self.freq).sin());
        if sines < 0.0 {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}

struct ImageTexture {
    pixels: Vec<Color>,
    width: usize,
    height: usize,
}

impl ImageTexture {
    fn new(path: &str) -> Self {
        let rgbimg = image::open(path).unwrap().to_rgb8();
        let (w, h) = rgbimg.dimensions();
        let mut image = vec![Color::zero(); (w * h) as usize];
        for(i, (_, _, pixel)) in image.iter_mut().zip(rgbimg.enumerate_pixels()) {
            *i = Color::from_rgb(pixel[0], pixel[1], pixel[2])
        }
        Self { pixels: image, width: w as usize, height: h as usize }
    }

    fn sample(&self, u: i64, v: i64) -> Color {
        let tu = if u < 0 { 0 }
            else if u as usize >= self.width { self.width - 1 }
            else { u as usize };
        let tv = if v < 0 { 0 }
            else if v as usize >= self.height { self.height - 1 }
            else { v as usize };
        self.pixels[tu + self.width * tv]
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, _p: Point3) -> Color {
        let x = (u * self.width as f64) as i64;
        let y = ((1.0 - v) * self.height as f64) as i64;
        self.sample(x, y)
    }
}

struct HitInfo {
    t: f64,
    p: Point3,
    n: Vec3,
    m: Arc<dyn Material>,
    u: f64, 
    v: f64
}

impl HitInfo {
    fn new(t: f64, p: Point3, n: Vec3, m: Arc<dyn Material>, u: f64, v: f64) -> Self {
        Self { t, p, n, m, u, v }
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
    albedo: Box<dyn Texture>,
}

impl Lambertian {
    fn new(albedo: Box<dyn Texture>) -> Self {
        Self { albedo }
    }
}

struct Metal {
    albedo: Box<dyn Texture>,
    fuzz: f64
}

struct Dielectric {
    ri: f64
}

impl Dielectric {
    const fn new (ri: f64) -> Self {
        Self { ri }
    }
    fn schlick(cosine: f64, ri: f64) -> f64 {
        let r0 = ((1.0 - ri) / (1.0 + ri)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Metal {
    fn new(albedo: Box<dyn Texture>, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let target = hit.p + hit.n + Vec3::random_in_unit_sphere();
        let albedo = self.albedo.value(hit.u, hit.v, hit.p);
        Some(ScatterInfo::new(Ray::new(hit.p, target - hit.p), albedo))
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let mut reflected = ray.direction.normalize().reflect(hit.n);
        reflected = reflected + self.fuzz * Vec3::random_in_unit_sphere();
        // 反射ベクトルと法線の角度が0より大きいときにscatterInfoを返す
        if reflected.dot(hit.n) > 0.0 {
            let albedo = self.albedo.value(hit.u, hit.v, hit.p);
            Some(ScatterInfo::new(Ray::new(hit.p, reflected), albedo))
        } else {
            None
        }
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let reflected = ray.direction.reflect(hit.n);
        let (outward_normal, ni_over_nt, cosine) = {
            let dot = ray.direction.dot(hit.n);
            if dot > 0.0 {
                (-hit.n, self.ri, self.ri * dot / ray.direction.length())
            } else {
                (hit.n, self.ri.recip(), -dot / ray.direction.length())
            }
        };

        if let Some(refracted) = (-ray.direction).refract(outward_normal, ni_over_nt) {
            if Vec3::random_full().x() > Self::schlick(cosine, self.ri) {
                return Some(ScatterInfo::new(Ray::new(hit.p, refracted), Color::one()));
            }
        }

        Some(ScatterInfo::new(Ray::new(hit.p, reflected), Color::one()))
    }
}

struct ShapeBuilder {
    texture: Option<Box<dyn Texture>>,
    material: Option<Arc<dyn Material>>,
    shape: Option<Box<dyn Shape>>
}

impl ShapeBuilder {
    fn new() -> Self {
        Self { texture: None, material: None, shape: None }
    }

    // load texture

    fn image_texture(mut self, path: &str) -> Self {
        self.texture = Some(Box::new(ImageTexture::new(path)));
        self
    }

    // textures

    fn color_texture(mut self, color: Color) -> Self {
        self.texture = Some(Box::new(ColorTexture::new(color)));
        self
    }

    fn checker_texture(mut self, odd_color: Color, even_color: Color, freq: f64) -> Self {
        self.texture = Some(Box::new(CheckerTexture::new(
            Box::new(ColorTexture::new(odd_color)),
            Box::new(ColorTexture::new(even_color)),
            freq
        )));
        self
    }

    // materials

    fn lambertian(mut self) -> Self {
        self.material = Some(Arc::new(Lambertian::new(self.texture.unwrap())));
        self.texture = None;
        self
    }

    fn metal(mut self, fuzz: f64) -> Self {
        self.material = Some(Arc::new(Metal::new(self.texture.unwrap(), fuzz)));
        self.texture = None;
        self
    }

    fn dieletric(mut self, ri: f64) -> Self {
        self.material = Some(Arc::new(Dielectric::new(ri)));
        self
    }

    // shpaes

    fn sphere(mut self, center: Point3, radius: f64) -> Self {
        self.shape = Some(Box::new(Sphere::new(center, radius, self.material.unwrap())));
        self.material = None;
        self
    }

    // build

    fn build(self) -> Box<dyn Shape> {
        self.shape.unwrap()
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

    // sphere mapping
    fn uv(p: Point3) -> (f64, f64) {
        let phi = p.z().atan2(p.x());
        let theta = p.y().asin();
        (1.0 - (phi + PI) / (2.0 * PI), (theta + PI / 2.0) / PI)
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
                let n = (p - self.center) / self.radius;
                let (u, v) = Self::uv(n);
                return Some(HitInfo::new(
                    temp,
                    p,
                    (p - self.center) / self.radius,
                    Arc::clone(&self.material),
                    u,
                    v
                ));
            }
            let temp = (-b + root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                let n = (p - self.center) / self.radius;
                let (u, v) = Self::uv(n);
                return Some(HitInfo::new(
                    temp,
                    p,
                    (p - self.center) / self.radius,
                    Arc::clone(&self.material),
                    u,
                    v
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

struct RandomScene {
    world: ShapeList
}

impl RandomScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        // world.push(ShapeBuilder::new()
        //     .lambertian(Color::new(0.5, 0.5, 0.5))
        //     .sphere(Point3::new(0.0, -1000.0, 0.0), 1000.0)
        //     .build()
        // );
        // for au in -11..11 {
        //     let a = au as f64;
        //     for bu in -11..11 {
        //         let b = bu as f64;
        //         let [rx, rz, material_choice] = Float3::random().to_array();
        //         let center = Point3::new(a + 0.9 * rx, 0.2, b + 0.9 * rz);
        //         if(center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
        //             world.push({
        //                 if material_choice < 0.8 {
        //                     let albedo = Color::random() * Color::random();
        //                     ShapeBuilder::new()
        //                         .lambertian(albedo)
        //                         .sphere(center, 0.2)
        //                         .build()
        //                 } else if material_choice < 0.95 {
        //                     let albedo = Color::random_limit(0.5, 1.0);
        //                     let fuzz = Float3::random_full().x();
        //                     ShapeBuilder::new()
        //                         .metal(albedo, fuzz)
        //                         .sphere(center, 0.2)
        //                         .build()
        //                 } else {
        //                     ShapeBuilder::new()
        //                         .dieletric(1.5)
        //                         .sphere(center, 0.2)
        //                         .build()
        //                 }
        //             })
        //         }
        //     }
        // }

        // world.push(ShapeBuilder::new()
        //     .dieletric(1.5)
        //     .sphere(Point3::new(0.0, 1.0, 0.0), 1.0)
        //     .build()
        // );
        // world.push(ShapeBuilder::new()
        //     .lambertian(Color::new(0.4, 0.2, 0.1))
        //     .sphere(Point3::new(-4.0, 1.0, 0.0), 1.0)
        //     .build()
        // );
        // world.push(ShapeBuilder::new()
        //     .metal(Color::new(0.7, 0.6, 0.5), 0.0)
        //     .sphere(Point3::new(4.0, 1.0, 0.0), 1.0)
        //     .build()
        // );

        Self { world }
    }

    fn background(&self, d: Vec3) -> Color {
        let t = 0.5 * (d.normalize().y() + 1.0);
        Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
    }
}

impl SceneWidthDepth for RandomScene {
    fn camera(&self) -> Camera {
        Camera::from_lookat(
            Point3::new(13.0, 2.0, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Vec3::yaxis(),
            20.0,
            self.aspect()
        )
    }
    fn trace(&self, ray: Ray, depth: usize) -> Color {
        let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        if let Some(hit) = hit_info {
            let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
            if let Some(scatter) = scatter_info {
                scatter.albedo * self.trace(scatter.ray, depth - 1)
            } else {
                Color::zero()
            }
        } else {
            self.background(ray.direction)
        }
    }
}

struct SimpleScene {
    world: ShapeList,
}

impl SimpleScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        // world.push(Box::new(Sphere::new(
        //     Point3::new(0.6, 0.0, -1.0),
        //     0.5,
        //     Arc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)))
        // )));
        // world.push(Box::new(Sphere::new(
        //     Point3::new(-0.6, 0.0, -1.0),
        //     0.5,
        //     Arc::new(Dielectric::new(1.5))
        // )));
        // world.push(Box::new(Sphere::new(
        //     Point3::new(-0.6, 0.0, -1.0),
        //     -0.45,
        //     Arc::new(Dielectric::new(1.5))
        // )));
        // world.push(Box::new(Sphere::new(
        //     Point3::new(-0.0, -0.35, -0.8),
        //     0.15,
        //     Arc::new(Metal::new(Color::new(0.8, 0.8, 0.8), 0.2))
        // )));
        // world.push(Box::new(Sphere::new(
        //     Point3::new(0.0, -100.5, -1.0),
        //     100.0,
        //     Arc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)))
        // )));
        world.push(ShapeBuilder::new()
            // .color_texture(Color::new(0.1, 0.2, 0.5))
            .image_texture("resources/brick_diffuse.jpg")
            .lambertian()
            .sphere(Point3::new(0.6, 0.0, -1.0), 0.5)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(Color::new(0.8, 0.8, 0.8))
            .metal(0.4)
            .sphere(Point3::new(-0.6, 0.0, -1.0), 0.5)
            .build()
        );
        world.push(ShapeBuilder::new()
            .checker_texture(
                Color::new(0.8, 0.8, 0.0),
                Color::new(0.8, 0.2, 0.0),
                10.0
            )
            .lambertian()
            .sphere(Point3::new(0.0, -100.5, -1.0), 100.0)
            .build()
        );
        Self { world }
    }
    fn background(&self, d: Vec3) -> Color {
        let t = 0.5 * (d.y() + 1.0);
        Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
    }
}

impl SceneWidthDepth for SimpleScene {
    fn camera(&self) -> Camera {
        Camera::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-2.0, -1.0, -1.0),
        )
    }

    fn trace(&self, ray: Ray, depth: usize) -> Color {
        let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        if let Some(hit) = hit_info {
            let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
            if let Some(scatter) = scatter_info {
                scatter.albedo * self.trace(scatter.ray, depth - 1)
            } else {
                Color::zero()
            }
        } else {
            self.background(ray.direction)
        }
    }
}
pub fn run() {
    // render(SimpleScene::new());
    render_aa_with_depth(SimpleScene::new());
    // render_aa_with_depth(RandomScene::new());
}

