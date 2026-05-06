#pragma once

namespace foundation {

// RAII class to limit the number of malloc arenas on Linux/glibc
// This helps reduce memory fragmentation when using many OpenMP threads.
// Does nothing on non-Linux platforms as their allocation are supposedly
// smarter than glibc's malloc.
class ScopedMallocArena {
 public:
  ScopedMallocArena();
  ~ScopedMallocArena();
};

}  // namespace foundation
