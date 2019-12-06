struct BTreeNode<K, V> {
    id: usize,
    elements: [(K, V); 511],
    pointers: [usize; 512]
}