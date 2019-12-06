const MAX_ARRAY_SIZE: usize = 16;

#[derive(Default)]
pub struct ArrayVec<T> {
    pub data: [T; MAX_ARRAY_SIZE],
    pub length: usize,
}

impl<'a, A: 'a> Extend<&'a A> for ArrayVec<A> where A: Copy {
    fn extend<T: IntoIterator<Item=&'a A>>(&mut self, iter: T) {
        for x in iter.into_iter() {
            self.data[self.length] = *x;
            self.length += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::array_vec::ArrayVec;

    #[test]
    fn test_extend() {
        let mut av = ArrayVec::default();
        av.extend(&[1, 2, 3, 4]);
        assert_eq!(av.length, 4);
        assert_eq!(av.data[0..5], [1, 2, 3, 4, 0]);
    }
}