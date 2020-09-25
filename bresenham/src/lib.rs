#![feature(const_generics)]
#![feature(const_generic_impls_guard)]

use num_traits::NumAssignRef;

pub type PointD<T, const D: usize> = [T; D];
pub type LineD<T, const D: usize> = (PointD<T, D>, PointD<T, D>);

pub trait Rasterize {
    type Iterator: core::iter::Iterator;
    fn rasterize(self) -> Self::Iterator;
}
pub trait RasterizeTwo<T, const D: usize> {
    fn rasterize_two<F: FnMut(&PointD<T, D>)>(&self, f: F);
}

impl<const D: usize> Rasterize for PointD<i32, D> {
    type Iterator = core::iter::Once<Self>;
    fn rasterize(self) -> Self::Iterator {
        core::iter::once(self)
    }
}

impl<const D: usize> RasterizeTwo<i32, D> for LineD<i32, D> {
    #[inline(always)]
    fn rasterize_two<F: FnMut(&PointD<i32, D>)>(&self, mut f: F) {
        let start = &self.0;
        let end = &self.1;

        let mut dist = array::map(&start, |d, s| helpers::diff(s, &end[d]));
        let swap = if dist[0] == 0 || true {
            dist.iter()
                .enumerate()
                .filter(|(_, di)| **di > 0)
                .min_by_key(|(_, d)| *d)
                .map(|(d, _)| d)
                .or(Some(0))
                .unwrap()
        } else {
            0
        };
        dist.swap(0, swap);
        let mut steps = 1;
        for d in dist.iter() {
            steps += d;
        }

        //let mut error = array::new::<_, _, D>(|_| 3 * dist[0]);
        let mut next_error_x = array::new::<_, _, D>(|d| 3 * dist[0] + dist[d]);
        let mut sign = array::new::<_, _, D>(|d| (end[d] - start[d]).signum());
        sign.swap(0, swap);

        let mut current = *start;

        for _ in 0..steps {
            f(&current);

            //compute which dimension would have the biggest error term after incrementing x
            let mut max_error_x = 1;
            for d in 2..D {
                if next_error_x[d] > next_error_x[max_error_x] {
                    max_error_x = d;
                }
            }
            debug_assert_ne!(max_error_x, 0);

            if next_error_x[max_error_x] - dist[max_error_x] <= dist[0] * 3 {
                current[swap] += sign[0];
                for d in 0..D {
                    next_error_x[d] += dist[d];
                }
            } else {
                if max_error_x == swap {
                    current[0] += sign[max_error_x];
                } else {
                    current[max_error_x] += sign[max_error_x];
                }
                next_error_x[max_error_x] -= dist[0];
            }
        }
    }
}

impl<const D: usize> Rasterize for LineD<i32, D> {
    type Iterator = BresenhamIterator<i32, D>;
    fn rasterize(self) -> Self::Iterator {
        Self::Iterator::new(self.0, self.1)
    }
}

pub struct BresenhamIterator<T, const D: usize> {
    current: PointD<T, D>,
    //end: PointD<T, D>,
    dist: PointD<T, D>,
    //min_dist_d: usize,
    swap: usize,
    error: PointD<T, D>,
    next_error_x: PointD<T, D>,
    sign: PointD<T, D>,
    steps: u64,
}

impl<const D: usize> BresenhamIterator<i32, D> {
    fn new(mut start: PointD<i32, D>, mut end: PointD<i32, D>) -> Self {
        let mut dist = array::new(|d| helpers::diff(&start[d], &end[d]));
        let swap = if dist[0] == 0 || true {
            dist.iter()
                .enumerate()
                .filter(|(_, di)| **di > 0)
                .min_by_key(|(_, d)| *d)
                .map(|(d, _)| d)
                .or(Some(0))
                .unwrap()
        } else {
            0
        };
        dist.swap(0, swap);
        start.swap(0, swap);
        end.swap(0, swap);
        let mut steps = 1;
        for d in dist.iter() {
            steps += d;
        }
        let mut current = start;
        current.swap(0, swap);
        Self {
            current: current,
            //end: end,
            dist: dist,
            //min_dist_d: min_dist_d,
            swap: swap,
            error: array::new(|_| 3 * dist[0]),
            next_error_x: array::new(|d| 3 * dist[0] + dist[d]),
            sign: array::new(|d| if start[d] < end[d] { 1 } else { -1 }),
            steps: steps as u64,
        }
    }
}

impl<const D: usize> Iterator for BresenhamIterator<i32, D> {
    type Item = PointD<i32, D>;
    //#[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.steps == 0 {
            return None;
        } else {
            self.steps -= 1;
        }
        let mut ret = self.current;

        //compute which dimension would have the biggest error term after incrementing x
        let max_error_x = self
            .next_error_x
            .iter()
            .enumerate()
            //.filter(|(d, _)| *d != self.min_dist_d)
            .skip(1)
            //.rev()
            .max_by_key(|(_, e)| *e)
            .unwrap()
            .0;
        //debug_assert_ne!(max_error_x, self.min_dist_d);
        debug_assert_ne!(max_error_x, 0);

        let mut e_d = self.error[max_error_x];
        //e_d -= &self.dist[self.min_dist_d];
        e_d -= &self.dist[0];

        if e_d <= self.dist[0] * 2 || self.next_error_x[max_error_x] < e_d {
            self.current[self.swap] += &self.sign[0];
            for d in 0..D {
                self.error[d] = self.next_error_x[d];
            }
            for d in 0..D {
                self.next_error_x[d] += self.dist[d];
            }
        } else {
            if max_error_x == self.swap || max_error_x == 0 {
                self.current[self.swap - max_error_x] += &self.sign[max_error_x];
            } else {
                self.current[max_error_x] += &self.sign[max_error_x];
            }
            let error = self.error[max_error_x] - self.dist[0];
            self.error[max_error_x] = error;
            self.next_error_x[max_error_x] = error + self.dist[max_error_x];
        }
        Some(ret)
    }
}

mod helpers {
    pub fn diff<T>(a: &T, b: &T) -> T
    where
        T: Ord + Clone + num_traits::NumAssignRef,
    {
        if a < b {
            let mut temp = b.clone();
            temp -= a;
            temp
        } else {
            let mut temp = a.clone();
            temp -= b;
            temp
        }
    }
    pub fn abs<T>(a: &T) -> T
    where
        T: Ord + Clone + num_traits::NumAssignRef,
    {
        if a < &T::zero() {
            let mut temp = T::zero();
            temp -= a;
            temp
        } else {
            a.clone()
        }
    }
}
pub mod array {
    /*
    Some functions to help working with const_generic arrays.
    */
    use std::mem::MaybeUninit;
    //create an array with const_generic parameter N as length
    pub fn new<F, T, const N: usize>(mut f: F) -> [T; N]
    where
        F: FnMut(usize) -> T,
    {
        // Create an explicitly uninitialized reference. The compiler knows that data inside
        // a `MaybeUninit<T>` may be invalid, and hence this is not UB:
        let mut x = MaybeUninit::<[T; N]>::uninit();
        // Set it to a valid value.
        for i in 0..N {
            unsafe {
                let ptr = x.as_mut_ptr() as *mut T;
                ptr.add(i).write(f(i));
            }
        }
        // Extract the initialized data -- this is only allowed *after* properly
        // initializing `x`!
        let x = unsafe { x.assume_init() };
        x
    }

    pub fn map<F, T, B, const N: usize>(a: &[T; N], mut f: F) -> [B; N]
    where
        F: FnMut(usize, &T) -> B,
    {
        new(|i| f(i, &a[i]))
    }

    pub fn map_mut<F, T, B, const N: usize>(a: &mut [T; N], mut f: F) -> [B; N]
    where
        F: FnMut(&mut T) -> B,
    {
        new(|i| f(&mut a[i]))
    }

    pub fn zip_map<F, T, U, B, const N: usize>(a: &[T; N], b: &[U; N], mut f: F) -> [B; N]
    where
        F: FnMut(&T, &U) -> B,
    {
        new(|i| {
            let ret = f(&a[i], &b[i]);
            ret
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_2D() {
        const size: usize = 21;
        let test = |line: LineD<i32, 2>| {
            let mut draw = [[' '; size]; size];
            let mut points = Vec::new();
            line.rasterize_two(|p| {
                draw[p[1] as usize][p[0] as usize] = '+';
                points.push(*p);
            });
            assert_eq!(points[0], line.0);
            assert_eq!(points.last().unwrap(), &line.1);
            for row in draw.iter() {
                for c in row.iter() {
                    print!("{}", c);
                }
                println!("");
            }
        };

        let line = ([10, 10], [20, 10]);
        test(line);
        let line = ([10, 10], [10, 20]);
        test(line);
        let line = ([10, 10], [0, 10]);
        test(line);
        let line = ([10, 10], [10, 0]);
        test(line);
    }
    #[test]
    fn test_simple_3D() {
        let line = ([0, 0, 0], [2, 2, 2]);
        println!("{:?}", line);
        for p in line.rasterize() {
            println!("{:?}", p);
        }
    }
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
