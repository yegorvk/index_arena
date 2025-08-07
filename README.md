# index_arena

A simple, id-based, heterogeneous arena allocator.

## Id-based

Uses small unique identifiers instead of references to represent allocations.
This leverages the type system to statically assign every identifier to the
arena it belongs to, ensuring safety without incurring runtime overhead.

Accessing individual elements is done through the various
arena methods, conceptually similar to indexing a `Vec`.

## Heterogeneous

Supports allocating values of all `Sized` types as well slices and strings.
This is particularly useful in scenarios where you have tree-like data
structures with different node types.

## Statically guaranteed safety

The implementation leverages the power of the Rust's type
system, achieving safety with almost no runtime checks.

## No `Drop`

Due to the way this crate works, the arena cannot track individual allocations,
so it doesn't drop its elements. This is a necessary trade off between
memory usage and element tracking.

## Examples

```rust
use index_arena::{Id, new_arena};

struct Even<A> {
     next: Option<Id<Odd<A>, A>>,
}

struct Odd<A> {
     next: Option<Id<Even<A>, A>>,
}

let mut arena = new_arena!();

let three = arena.alloc(Odd { next: None });
let two = arena.alloc(Even { next: Some(three) });
let one = arena.alloc(Odd { next: Some(two) });

assert_eq!(&arena[one].next, &Some(two));
```
