// Contains all aggregate functions with their
// structs (data types) that contain their internal state.
//
// All aggregate functions implement one trait called "Aggregate"
// which ensure that all of them have the same interface.

/// Basic trait used for all aggregate functions.
pub trait Aggregate<T> {
    fn next(&mut self, item: T);
    fn result(&mut self) -> T;
}

/// Aggregate function that computes arithmetic mean of values.
pub struct Average {
    avg: f32,
    count: u64,
}

impl Aggregate<f32> for Average {
    fn next(&mut self, item: f32) {
        self.avg += (item - self.avg) / self.count as f32;
        self.count += 1;
    }

    fn result(&mut self) -> f32 {
        self.avg
    }
}

impl Default for Average {
    fn default() -> Self {
        Average { avg: 0.0, count: 1 }
    }
}

/// Aggregate function that counts number of items.
#[derive(Default)]
pub struct Count(usize);

impl Aggregate<f32> for Count {
    fn next(&mut self, _: f32) {
        self.0 += 1;
    }

    fn result(&mut self) -> f32 {
        self.0 as f32
    }
}

/// Generates implementation of Aggregate for specified Type.
///
/// # Examples
/// ```
/// impl_basic_math_aggregate!(Min, min);
/// ```
macro_rules! impl_basic_math_aggregate {
    ($agg_name:ident, $fn_name:ident) => {
        /// Aggregate function that finds `$fn_name` value of items.
        #[derive(Default)]
        pub struct $agg_name(f32);

        impl Aggregate<f32> for $agg_name {
            fn next(&mut self, item: f32) {
                self.0 = (self.0).$fn_name(item);
            }

            fn result(&mut self) -> f32 {
                self.0
            }
        }
    };
}

// implement Min and Max aggregates
impl_basic_math_aggregate!(Min, min);
impl_basic_math_aggregate!(Max, max);

/// Aggregate function that sums (adds together) items.
#[derive(Default)]
pub struct Sum(f32);

impl Aggregate<f32> for Sum {
    fn next(&mut self, item: f32) {
        self.0 += item;
    }

    fn result(&mut self) -> f32 {
        self.0
    }
}