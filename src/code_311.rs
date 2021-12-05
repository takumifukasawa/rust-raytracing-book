
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
    pdf: Option<Arc<dyn Pdf>>
}

impl ScatterInfo {
    fn new(ray: Ray, albedo: Color, pdf: Option<Arc<dyn Pdf>>) -> Self {
        Self { ray, albedo, pdf }
    }
}

// Send ... 所有権を移動できることの明示
trait Material: Sync + Send {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo>;
    fn emitted(&self, _ray: &Ray, _hit: &HitInfo) -> Color { Color::zero() }
    fn scattering_pdf(&self, _ray: &Ray, _hit: &HitInfo) -> f64 { 0.0 }
}

struct Lambertian {
    albedo: Box<dyn Texture>,
    pdf: Arc<dyn Pdf>
}

impl Lambertian {
    fn new(albedo: Box<dyn Texture>) -> Self {
        Self { albedo, pdf: Arc::new(CosinePdf::new()) }
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

struct DiffuseLight {
    emit: Box<dyn Texture>,
}

impl DiffuseLight {
    fn new(emit: Box<dyn Texture>) -> Self {
        Self { emit }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        // let direction = ONB::new(hit.n).local(Vec3::random_cosine_direction());
        // let new_ray = Ray::new(hit.p, direction.normalize());
        // let albedo = self.albedo.value(hit.u, hit.v, hit.p);
        // let pdf_value = new_ray.direction.dot(hit.n) * FRAC_1_PI;
        // Some(ScatterInfo::new(new_ray, albedo, pdf_value))
        let albedo = self.albedo.value(hit.u, hit.v, hit.p);
        Some(ScatterInfo::new(*ray, albedo, Some(Arc::clone(&self.pdf))))
    }
    fn scattering_pdf(&self, ray: &Ray, hit: &HitInfo) -> f64 {
        ray.direction.normalize().dot(hit.n).max(0.0) * FRAC_1_PI
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let mut reflected = ray.direction.normalize().reflect(hit.n);
        reflected = reflected + self.fuzz * Vec3::random_in_unit_sphere();
        // 反射ベクトルと法線の角度が0より大きいときにscatterInfoを返す
        if reflected.dot(hit.n) > 0.0 {
            let albedo = self.albedo.value(hit.u, hit.v, hit.p);
            // Some(ScatterInfo::new(Ray::new(hit.p, reflected), albedo, 0.0))
            Some(ScatterInfo::new(Ray::new(hit.p, reflected), albedo, None))
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
                return Some(ScatterInfo::new(Ray::new(hit.p, refracted), Color::one(), None));
            }
        }

        Some(ScatterInfo::new(Ray::new(hit.p, reflected), Color::one(), None))
    }
}

impl Material for DiffuseLight {
    fn scatter(&self, _ray: &Ray, _hit: &HitInfo) -> Option<ScatterInfo> {
        None
    }
    fn emitted(&self, ray: &Ray, hit: &HitInfo) -> Color {
        if ray.direction.dot(hit.n) < 0.0 {
            self.emit.value(hit.u, hit.v, hit.p)
        } else {
            Color::zero()
        }
    }
}

trait Pdf: Send + Sync {
    fn value(&self, hit: &HitInfo, direction: Vec3) -> f64;
    fn generate(&self, hit: &HitInfo) -> Vec3;
}

struct CosinePdf {}

impl CosinePdf {
    const fn new() -> Self {
        Self {}
    }
}

impl Pdf for CosinePdf {
    fn value(&self, hit: &HitInfo, direction: Vec3) -> f64 {
        let cosine = direction.normalize().dot(hit.n);
        if cosine > 0.0 {
            cosine * FRAC_1_PI
        } else {
            0.0
        }
    }

    fn generate(&self, hit: &HitInfo) -> Vec3 {
        ONB::new(hit.n).local(Vec3::random_cosine_direction())
    }
}

struct ShapePdf {
    shape: Arc<dyn Shape>,
    origin: Point3
}

impl ShapePdf {
    fn new(shape: Arc<dyn Shape>, origin: Point3) -> Self {
        Self { shape, origin }
    }
}

impl Pdf for ShapePdf {
    fn value(&self, _hit: &HitInfo, direction: Vec3) -> f64 {
        self.shape.pdf_value(self.origin, direction)
    }

    fn generate(&self, _hit: &HitInfo) -> Vec3 {
        self.shape.random(self.origin)
    }
}

struct MixturePdf {
    // pdfs: [Box<dyn Pdf>; 2]
    pdfs: [Arc<dyn Pdf>; 2]
}

impl MixturePdf {
    // fn new(pdf0: Box<dyn Pdf>, pdf1: Box<dyn Pdf>) -> Self {
    fn new(pdf0: Arc<dyn Pdf>, pdf1: Arc<dyn Pdf>) -> Self {
        Self { pdfs: [pdf0, pdf1] }
    }
}

impl Pdf for MixturePdf {
    fn value(&self, hit: &HitInfo, direction: Vec3) -> f64 {
        let pdf0 = self.pdfs[0].value(&hit, direction);
        let pdf1 = self.pdfs[1].value(&hit, direction);
        0.5 * pdf0 + 0.5 * pdf1
    }
    fn generate(&self, hit: &HitInfo) -> Vec3 {
        if Vec3::random_full().x() < 0.5 {
            self.pdfs[0].generate(hit)
        } else {
            self.pdfs[1].generate(hit)
        }
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

    // affine

    fn translate(mut self, offset: Point3) -> Self {
        self.shape = Some(Box::new(Translate::new(self.shape.unwrap(), offset)));
        self
    }

    fn rotate(mut self, axis: Vec3, angle: f64) -> Self {
        self.shape = Some(Box::new(Rotate::new(self.shape.unwrap(), axis, angle)));
        self
    }

    // material light

    fn diffuse_light(mut self) -> Self {
        self.material = Some(Arc::new(DiffuseLight::new(self.texture.unwrap())));
        self.texture = None;
        self
    }

    // rect

    fn rect_xy(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::XY, self.material.unwrap())));
        self.material = None;
        self
    }

    fn rect_xz(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::XZ, self.material.unwrap())));
        self.material = None;
        self
    }

    fn rect_yz(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::YZ, self.material.unwrap())));
        self.material = None;
        self
    }

    // decorators

    fn flip_face(mut self) -> Self {
        self.shape = Some(Box::new(FlipFace::new(self.shape.unwrap())));
        self
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

    fn material(mut self, material: Arc<dyn Material>) -> Self {
        self.material = Some(material);
        self.texture = None;
        self
    }

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

    fn box3d(mut self, p0: Point3, p1: Point3) -> Self {
        self.shape = Some(Box::new(Box3D::new(p0, p1, self.material.unwrap())));
        self.material = None;
        self
    }

    // build

    fn build(self) -> Box<dyn Shape> {
        self.shape.unwrap()
    }
}

// Sync ... 複数スレッドから参照されても安全なことを意味する
trait Shape: Send + Sync {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo>;
    fn pdf_value(&self, _o: Vec3, _v: Vec3) -> f64 { 0.0 }
    fn random(&self, _o: Vec3) -> Vec3 { Vec3::xaxis() }
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

    fn pdf_value(&self, o: Vec3, v: Vec3) -> f64 {
        if let Some(_) = self.hit(&Ray::new(o, v), 0.001, f64::MAX) {
            let dd = (self.center - o).length_squared();
            let rr = self.radius.powi(2).min(dd);
            let cos_theta_max = (1.0 - rr * dd.recip()).sqrt();
            let solid_angle = PI2 * (1.0 - cos_theta_max);
            solid_angle.recip()
        } else {
            0.0
        }
    }

    fn random(&self, o: Vec3) -> Vec3 {
        let direction = self.center - o;
        let distance_squared = direction.length_squared();
        ONB::new(direction).local(Vec3::random_to_sphere(self.radius, distance_squared))
    }
}

enum RectAxisType {
    XY,
    XZ,
    YZ
}

struct Rect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    k: f64,
    axis: RectAxisType,
    material: Arc<dyn Material>
}

impl Rect {
    fn new(x0: f64, x1: f64, y0: f64, y1: f64, k: f64, axis: RectAxisType, material: Arc<dyn Material>) -> Self {
        Self { x0, x1, y0, y1, k, axis, material }
    }
}

impl Shape for Rect {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let mut origin = ray.origin;
        let mut direction = ray.direction;
        let mut axis = Vec3::zaxis();
        match self.axis {
            RectAxisType::XY => {}
            RectAxisType::XZ => {
                origin = Point3::new(origin.x(), origin.z(), origin.y());
                direction = Vec3::new(direction.x(), direction.z(), direction.y());
                axis = Vec3::yaxis();
            }
            RectAxisType::YZ => {
                origin = Point3::new(origin.y(), origin.z(), origin.x());
                direction = Vec3::new(direction.y(), direction.z(), direction.x());
                axis = Vec3::xaxis();
            }
        }

        let t = (self.k - origin.z()) / direction.z();
        if t < t0 || t > t1 {
            return None;
        }

        let x = origin.x() + t * direction.x();
        let y = origin.y() + t * direction.y();

        if x < self.x0 || x > self.x1 || y <self.y0 || y > self.y1 {
            return None;
        }

        Some(HitInfo::new(
            t,
            ray.at(t),
            axis,
            Arc::clone(&self.material),
            (x - self.x0) / (self.x1 - self.x0),
            (y - self.y0) / (self.y1 - self.y0)
        ))
    }

    fn pdf_value(&self, o: Vec3, v: Vec3) -> f64 {
        if let Some(hit) = self.hit(&Ray::new(o, v), 0.001, f64::MAX) {
            let area = (self.x1 - self.x0) * (self.y1 - self.y0);
            let distance_squared = hit.t.powi(2) * v.length_squared();
            let cosine = v.dot(hit.n).abs() / v.length();
            distance_squared / (cosine * area)
        } else {
            0.0
        }
    }

    fn random(&self, o: Vec3) -> Vec3 {
        let [rx, ry, _] = Vec3::random().to_array();
        let x = self.x0 + rx * (self.x1 - self.x0);
        let y = self.y0 + ry * (self.y1 - self.y0);
        match self.axis {
            RectAxisType::XY => Point3::new(x, y, self.k) - o,
            RectAxisType::XZ => Point3::new(x, self.k, y) - o,
            RectAxisType::YZ => Point3::new(self.k, x, y) - o,
        }
    }
}

struct FlipFace {
    shape: Box<dyn Shape>
}

impl FlipFace {
    fn new(shape: Box<dyn Shape>) -> Self {
        Self { shape }
    }
}

impl Shape for FlipFace {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        if let Some(hit) = self.shape.hit(ray, t0, t1) {
            Some(HitInfo { n: -hit.n, ..hit })
        } else {
            None
        }
    }
}

struct Box3D {
    p0: Point3,
    p1: Point3,
    shapes: ShapeList
}

impl Box3D {
    fn new(p0: Point3, p1: Point3, material: Arc<dyn Material>) -> Self {
        let mut shapes = ShapeList::new();
        shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_xy(p0.x(), p1.x(), p0.y(), p1.y(), p1.z())
            .build()
        );
        shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_xy(p0.x(), p1.x(), p0.y(), p1.y(), p0.z())
            .flip_face()
            .build()
        );
        shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_xz(p0.x(), p1.x(), p0.z(), p1.z(), p1.y())
            .build()
        );
        shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_xz(p0.x(), p1.x(), p0.z(), p1.z(), p0.y())
            .flip_face()
            .build()
        );
        shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_yz(p0.y(), p1.y(), p0.z(), p1.z(), p1.x())
            .build()
        );
         shapes.push(ShapeBuilder::new()
            .material(Arc::clone(&material))
            .rect_yz(p0.y(), p1.y(), p0.z(), p1.z(), p0.x())
            .flip_face()
            .build()
        );
 
        Self { p0, p1, shapes }
    }
}

impl Shape for Box3D {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        self.shapes.hit(ray, t0, t1)
    }
}

struct Translate {
    shape: Box<dyn Shape>,
    offset: Point3,
}

impl Translate {
    fn new(shape: Box<dyn Shape>, offset: Point3) -> Self {
        Self { shape, offset }
    }
}

impl Shape for Translate {
    fn hit (&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let moved_ray = Ray::new(ray.origin - self.offset, ray.direction);
        if let Some(hit) = self.shape.hit(&moved_ray, t0, t1) {
            Some(HitInfo { p: hit.p + self.offset, ..hit })
        } else {
            None
        }
    }
}

struct Rotate {
    shape: Box<dyn Shape>,
    quat: Quat,
}

impl Rotate {
    fn new(shape: Box<dyn Shape>, axis: Vec3, angle: f64) -> Self {
        Self { shape, quat: Quat::from_rot(axis, angle.to_radians()) }
    }
}

impl Shape for Rotate {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let revq = self.quat.conj();
        let rotated_ray = Ray::new(revq.rotate(ray.origin), revq.rotate(ray.direction));
        if let Some(hit) = self.shape.hit(&rotated_ray, t0, t1) {
            Some(HitInfo { p: self.quat.rotate(hit.p), n: self.quat.rotate(hit.n), ..hit })
        } else {
            None
        }
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

    fn pdf_value(&self, o: Vec3, v: Vec3) -> f64 {
        if self.objects.is_empty() { panic!(); }
        let weight = 1.0 / self.objects.len() as f64;
        self.objects.iter().fold(0.0, |acc, s| acc + weight * s.pdf_value(o, v))
    }

    fn random(&self, o: Vec3) -> Vec3 {
        if self.objects.is_empty() { panic!(); }
        let index = (Vec3::random_full().x() * self.objects.len() as f64).floor() as usize;
        self.objects[index].random(o)
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

        // world.push(ShapeBuilder::new()
        //     // .color_texture(Color::new(0.1, 0.2, 0.5))
        //     .image_texture("resources/brick_diffuse.jpg")
        //     .lambertian()
        //     .sphere(Point3::new(0.6, 0.0, -1.0), 0.5)
        //     .build()
        // );
        // world.push(ShapeBuilder::new()
        //     .color_texture(Color::new(0.8, 0.8, 0.8))
        //     .metal(0.4)
        //     .sphere(Point3::new(-0.6, 0.0, -1.0), 0.5)
        //     .build()
        // );
        // world.push(ShapeBuilder::new()
        //     .checker_texture(
        //         Color::new(0.8, 0.8, 0.0),
        //         Color::new(0.8, 0.2, 0.0),
        //         10.0
        //     )
        //     .lambertian()
        //     .sphere(Point3::new(0.0, -100.5, -1.0), 100.0)
        //     .build()
        // );

        world.push(ShapeBuilder::new()
            .color_texture(Color::full(0.5))
            .lambertian()
            .sphere(Point3::new(0.0, 2.0, 0.0), 2.0)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(Color::full(4.0))
            .diffuse_light()
            .rect_xy(3.0, 5.0, 1.0, 3.0, -2.0)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(Color::full(0.8))
            .lambertian()
            .sphere(Point3::new(0.0, -1000.0, 0.0), 1000.0)
            .build()
        );

        Self { world }
    }
    fn background(&self, _: Vec3) -> Color {
        // let t = 0.5 * (d.y() + 1.0);
        // Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
        Color::full(0.1)
    }
}

impl SceneWidthDepth for SimpleScene {
    fn camera(&self) -> Camera {
        Camera::from_lookat(
            Vec3::new(13.0, 2.0, 3.0),
            Vec3::yaxis(),
            Vec3::yaxis(),
            30.0,
            self.aspect()
        )
    }

    fn trace(&self, ray: Ray, depth: usize) -> Color {
        let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        if let Some(hit) = hit_info {
            let emitted = hit.m.emitted(&ray, &hit);
            let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
            if let Some(scatter) = scatter_info {
                emitted + scatter.albedo * self.trace(scatter.ray, depth - 1)
            } else {
                emitted
            }
        } else {
            self.background(ray.direction)
        }
    }
}

struct CornelBoxScene {
    world: ShapeList,
    light: Arc<dyn Shape>
}

impl CornelBoxScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        let red = Color::new(0.64, 0.05, 0.05);
        let white = Color::full(0.73);
        let green = Color::new(0.12, 0.45, 0.15);

        world.push(ShapeBuilder::new()
            .color_texture(green)
            .lambertian()
            .rect_yz(0.0, 555.0, 0.0, 555.0, 555.0)
            .flip_face()
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(red)
            .lambertian()
            .rect_yz(0.0, 555.0, 0.0, 555.0, 0.0)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(Color::full(15.0))
            .diffuse_light()
            .rect_xz(213.0, 343.0, 227.0, 332.0, 554.0)
            .flip_face()
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xz(0.0, 555.0, 0.0, 555.0, 555.0)
            .flip_face()
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xz(0.0, 555.0, 0.0, 555.0, 0.0)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xy(0.0, 555.0, 0.0, 555.0, 555.0)
            .flip_face()
            .build()
        );

        // world.push(ShapeBuilder::new()
        //     .color_texture(white)
        //     .lambertian()
        //     .box3d(Point3::zero(), Point3::full(165.0))
        //     .rotate(Vec3::yaxis(), -18.0)
        //     .translate(Point3::new(130.0, 0.0, 65.0))
        //     .build()
        // );
        // world.push(ShapeBuilder::new()
        //     .color_texture(white)
        //     .metal(0.0)
        //     .box3d(Point3::zero(), Point3::new(165.0, 330.0, 165.0))
        //     .rotate(Vec3::yaxis(), 15.0)
        //     .translate(Point3::new(265.0, 0.0, 295.0))
        //     .build()
        // );

        world.push(ShapeBuilder::new()
            .dieletric(1.5)
            .sphere(Point3::new(190.0, 90.0, 190.0), 90.0)
            .build()
        );
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .box3d(Point3::zero(), Point3::new(165.0, 330.0, 165.0))
            .rotate(Vec3::yaxis(), 15.0)
            .translate(Point3::new(265.0, 0.0, 295.0))
            .build()
        );

        let mut light = ShapeList::new();
        light.push(ShapeBuilder::new()
            .color_texture(Color::zero())
            .lambertian()
            .rect_xz(213.0, 343.0, 227.0, 332.0, 554.0)
            .build()
        );
        light.push(ShapeBuilder::new()
            .color_texture(Color::zero())
            .lambertian()
            .sphere(Point3::new(190.0, 90.0, 190.0), 90.0)
            .build()
        );

        Self { world, light: Arc::new(light) }
    }

    fn background(&self, _: Vec3) -> Color {
        Color::full(0.0)
    }
}

impl SceneWidthDepth for CornelBoxScene {
    fn camera(&self) -> Camera {
        Camera::from_lookat(
            Vec3::new(278.0, 278.0, -800.0),
            Vec3::new(278.0, 278.0, 0.0),
            Vec3::yaxis(),
            40.0,
            self.aspect()
        )
    }

    fn trace(&self, ray: Ray, depth: usize) -> Color {
        // let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        // if let Some(hit) = hit_info {
        //     let emitted = hit.m.emitted(&ray, &hit);
        //     let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
        //     if let Some(scatter) = scatter_info {
        //         let pdf_value = hit.m.scattering_pdf(&scatter.ray, &hit);
        //         let albedo = scatter.albedo * pdf_value;
        //         emitted + albedo * self.trace(scatter.ray, depth - 1) / scatter.pdf_value
        //     } else {
        //         emitted
        //     }
        // } else {
        //     self.background(ray.direction)
        // }

        //let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        //if let Some(hit) = hit_info {
        //    let emitted = hit.m.emitted(&ray, &hit);
        //    let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
        //    if let Some(scatter) = scatter_info {
        //        let [rx, rz, _] = Point3::random().to_array();
        //        let on_light = Point3::new(213.0 + rx * (343.0 - 213.0), 554.0, 227.0 + rz * (332.0 - 227.0));
        //        let to_light = on_light - hit.p;
        //        let distance_squared = to_light.length_squared();
        //        let to_light = to_light.normalize();
        //        if to_light.dot(hit.n) < 0.0 {
        //            return emitted;
        //        }
        //        let light_area = (343.0 - 213.0) * (332.0 - 217.0);
        //        let light_cosine = to_light.y().abs();
        //        if light_cosine < 0.000001 {
        //            return emitted;
        //        }

        //        let spdf_value = distance_squared / (light_cosine * light_area);
        //        let new_ray = Ray::new(hit.p, to_light);

        //        let pdf_value = hit.m.scattering_pdf(&new_ray, &hit);
        //        let albedo = scatter.albedo * pdf_value;
        //        emitted + albedo * self.trace(new_ray, depth - 1) / spdf_value
        //    } else {
        //        emitted
        //    }
        //} else {
        //    self.background(ray.direction)
        //}

        let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        if let Some(hit) = hit_info {
            let emitted = hit.m.emitted(&ray, &hit);
            let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
            if let Some(scatter)  = scatter_info {
                if let Some(pdf) = scatter.pdf {
                   // let pdf = MixturePdf::new(
                   //      Arc::new(ShapePdf::new(
                   //          ShapeBuilder::new()
                   //              .color_texture(Color::zero())
                   //              .lambertian()
                   //              .rect_xz(213.0, 343.0, 227.0, 332.0, 554.0)
                   //              .build(),
                   //          hit.p
                   //      )),
                   //      Arc::clone(&pdf)
                   //  );
                    let shape_pdf = Arc::new(ShapePdf::new(Arc::clone(&self.light), hit.p));
                    let pdf = MixturePdf::new(shape_pdf, Arc::clone(&pdf));
                    let new_ray = Ray::new(hit.p, pdf.generate(&hit));
                    let spdf_value = pdf.value(&hit, new_ray.direction);
                    if spdf_value > 0.0 {
                        let pdf_value = hit.m.scattering_pdf(&new_ray, &hit);
                        let albedo = scatter.albedo * pdf_value;
                        emitted + albedo * self.trace(new_ray, depth - 1) / spdf_value
                    } else {
                        emitted
                    }
                } else {
                    emitted + scatter.albedo * self.trace(scatter.ray, depth - 1)
                }
            } else {
                emitted
            }
        } else {
            self.background(ray.direction)
        }
    }

    fn width(&self) -> u32 { 200 }
    fn height(&self) -> u32 { 200 }
}


pub fn run() {
    // render(SimpleScene::new());
    // render_aa_with_depth(SimpleScene::new());
    // render_aa_with_depth(RandomScene::new());
    render_aa_with_depth(CornelBoxScene::new());
}

