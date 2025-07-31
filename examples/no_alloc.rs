// cargo run --example no_alloc --no-default-features

use core::mem::MaybeUninit;

use index_arena::Storage;
use static_cell::StaticCell;

struct SliceStorage<'a> {
    bytes: &'a mut [MaybeUninit<u8>],
    current_byte_offset: usize,
}

impl<'a> SliceStorage<'a> {
    fn new(bytes: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            bytes,
            current_byte_offset: 0,
        }
    }
}

impl<'a> Storage for SliceStorage<'a> {
    fn alloc_raw(&mut self, len: usize) -> Option<usize> {
        let old_byte_offset = self.current_byte_offset;
        let new_byte_offset = self.current_byte_offset + len;
        if new_byte_offset > self.bytes.len() {
            None
        } else {
            self.current_byte_offset = new_byte_offset;
            Some(old_byte_offset)
        }
    }

    fn current_byte_offset(&self) -> usize {
        self.current_byte_offset
    }

    fn as_ptr(&self) -> *const MaybeUninit<u8> {
        self.bytes.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut MaybeUninit<u8> {
        self.bytes.as_mut_ptr()
    }

    fn free_bytes(&self) -> Option<usize> {
        Some(self.bytes.len() - self.current_byte_offset)
    }
}

static STATIC_BUFFER: StaticCell<[MaybeUninit<u8>; 128]> = StaticCell::new();

fn main() {
    let static_buffer = STATIC_BUFFER.init([MaybeUninit::uninit(); 128]);
    let mut stack_buffer: [MaybeUninit<u8>; 128] = [MaybeUninit::uninit(); 128];

    let mut arena_in_static =
        index_arena::new_arena_with_storage!(SliceStorage::new(static_buffer));
    let mut arena_in_stack =
        index_arena::new_arena_with_storage!(SliceStorage::new(&mut stack_buffer));

    let hello = arena_in_static.alloc_str("Hello, world on static");
    println!("hello = \"{}\"", &arena_in_static[hello]);

    let hello_again = arena_in_static.try_alloc_str("Hello, world on static");
    println!("hello_again = {:?}", hello_again);

    let hello = arena_in_stack.alloc_str("Hello, world on stack");
    println!("hello = \"{}\"", &arena_in_stack[hello]);

    let hello_again = arena_in_stack.try_alloc_str("Hello, world on stack");
    println!("hello_again = {:?}", hello_again);
}
