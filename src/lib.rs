extern crate stl;

#[derive(PartialEq, Clone, Copy, Debug)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Line {
    p1: Point,
    p2: Point,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Plane {
    p: Point,
    n: Point,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum Shape {
    point(Point),
    line(Line),
    triangle(Triangle),
    plane(Plane),
}

impl Point {
    fn new (x: f32, y: f32, z: f32) -> Point{
        Point{x: x, y: y, z: z}
    }

    fn sum (&self) -> f32 {
       self.x + self.y + self.z
    }

    fn add (&self, other: &Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn sub (&self, other: &Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn mult (&self, other: &Point) -> Point {
        Point::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    fn dot (&self, other: &Point) -> f32 {
        self.mult(other).sum()
    }

    fn mult_float (&self, f: &f32) -> Point {
        Point::new(self.x * f, self.y * f, self.z * f)
    }

    fn cross (&self, other: &Point) -> Point {
        Point::new(self.y * other.z - self.z * other.y,
                   self.z * other.x - self.x * other.z,
                   self.x * other.y - self.y * other.x)
     }

     fn is_infinite(&self) -> bool {
         self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
     }

     fn is_nan(&self) -> bool {
        !self.is_infinite() && (self.x.is_nan() || self.y.is_nan() || self.z.is_nan())
     }

    fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

impl Line {
    fn new(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> Line {
        Line{
             p1: Point::new(p1.0, p1.1, p1.2),
             p2: Point::new(p2.0, p2.1, p2.2),
        }
    }

    fn from_points(p1: &Point, p2: &Point) -> Line {
        Line{
            p1: *p1,
            p2: *p2,
        }
    }

    fn on_line(&self, point: &Point) -> bool {
        self.p1.distance(point) + self.p2.distance(point) == self.p1.distance(&self.p2)
    }

    fn intersect(&self, plane: &Plane) -> Option<Shape> {
        let direction = self.p2.sub(&self.p1);
        let orthogonal = plane.n.dot(&direction);
        let w = self.p1.sub(&plane.p);
        let fac = (-(plane.n.dot(&w))) / orthogonal;
        let v = direction.mult_float(&fac);
        let answer = self.p1.add(&v);
        if answer.is_infinite() {
            None
        } else if answer.is_nan() {
            Some(Shape::line(*self))
        } else {
            Some(Shape::point(answer))
        }
    }  
}

impl Triangle {
    fn new(p1: (f32, f32, f32), p2: (f32, f32, f32), p3: (f32, f32, f32)) -> Triangle {
        Triangle {
            p1: Point::new(p1.0, p1.1, p1.2),
            p2: Point::new(p2.0, p2.1, p2.2),
            p3: Point::new(p3.0, p3.1, p3.2),
        }
    }

    fn intersect(&self, plane: &Plane) -> Option<Shape> {
        let lines = vec![Line::from_points(&self.p1, &self.p2),
                         Line::from_points(&self.p2, &self.p3),
                         Line::from_points(&self.p3, &self.p1)];
        let mut results: Vec<Shape> = Vec::new();
        for line in lines {
            if let Some(answer) = line.intersect(&plane) {
                results.push(answer);
            }
        }
        if results.len() == 2 {
            if let (Shape::point(p1), Shape::point(p2)) = (results[0], results[1]) {
                return Some(Shape::line(Line::from_points(&p1, &p2)))
            }
        }
        results.retain(|&x| x.is_line());
        match results.len() {
            0 => None,
            1 => Some(results[0]),
            _ => Some(Shape::triangle(*self)),
        }
    }
}

impl Plane {
    fn new(p: (f32, f32, f32), n: (f32, f32, f32)) -> Plane {
        Plane{
            p: Point::new(p.0, p.1, p.2),
            n: Point::new(n.0, n.1, n.2),
        }
    }
}

impl Shape {
    fn is_line(&self) ->  bool {
        match *self {
            Shape::line(_) => true,
            _ => false,
        }
    }
}

fn main() {
    let line = Line::new((0.0, 0.0, 0.0), (1.0, 4.0, 2.0));
    let plane = Plane::new((0.0, 0.0, 1.0), (0.0, 0.0, 2.0));
    let answer = line.intersect(&plane);
    println!("intersecting\n    {:?}", answer);
    let line = Line::new((0.0, 0.0, 0.0), (2.0, 2.0, 0.0));
    let answer = line.intersect(&plane);
    println!("not intersecting\n    {:?}", answer);
    let line = Line::new((0.0, 0.0, 1.0), (1.0, 1.0, 1.0));
    let answer = line.intersect(&plane);
    println!("on plane\n    {:?}", answer);
    let triangle = Triangle::new((0.0, 0.0, 0.0),
                                 (2.0, 2.0, 0.0),
                                 (1.0, 1.0, 3.0));
    let answer = triangle.intersect(&plane);
    println!("triangle intersecting\n    {:?}", answer);
    let triangle = Triangle::new((0.0, 0.0, 1.0),
                                 (1.0, 0.0, 1.0),
                                 (0.0, 0.0, 0.0));
    let answer = triangle.intersect(&plane);
    println!("triangle line on plane\n    {:?}", answer);
    let triangle = Triangle::new((0.0, 0.0, 0.0),
                                 (1.0, 0.0, 0.0),
                                 (0.0, 1.0, 0.0));
    let answer = triangle.intersect(&plane);
    println!("triangle not intersecting\n    {:?}", answer);
    let triangle = Triangle::new((0.0, 0.0, 1.0),
                                 (1.0, 0.0, 1.0),
                                 (0.0, 1.0, 1.0));
    let answer = triangle.intersect(&plane);
    println!("triangle on plane\n    {:?}", answer);
}
