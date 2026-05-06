#include <foundation/memory.h>

#ifdef __linux__
#include <malloc.h>
#endif

namespace foundation {

ScopedMallocArena::ScopedMallocArena() {
#ifdef __linux__
  // Limit to 2 arenas to prevent excessive fragmentation with many threads
  mallopt(M_ARENA_MAX, 2);
#endif
}

ScopedMallocArena::~ScopedMallocArena() {
#ifdef __linux__
  // 0 means default behavior (based M_ARENA_TEST)
  mallopt(M_ARENA_MAX, 0);
#endif
}

}  // namespace foundation
