#define _GNU_SOURCE

#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr std::size_t mmap_allocation_threshold = 0x10000;
constexpr std::size_t max_tracked_stacks = 16384;

struct stack_mapping {
    void* address;
    std::size_t length;
};

using new_array_fn = void* (*)(std::size_t);
using delete_array_fn = void (*)(void*);
using sized_delete_array_fn = void (*)(void*, std::size_t);

stack_mapping mappings[max_tracked_stacks]{};
std::size_t mapping_count = 0;
pthread_mutex_t mapping_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t symbols_once = PTHREAD_ONCE_INIT;
new_array_fn real_new_array = nullptr;
delete_array_fn real_delete_array = nullptr;
sized_delete_array_fn real_sized_delete_array = nullptr;
bool is_bundled_simulator = false;
bool allocation_message_written = false;

void write_message(const char* message)
{
    (void)write(STDERR_FILENO, message, std::strlen(message));
}

void detect_executable()
{
    char executable[4096];
    const ssize_t length =
        readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(executable) - 1)
        return;

    executable[length] = '\0';
    const char* name = std::strrchr(executable, '/');
    name = name == nullptr ? executable : name + 1;
    is_bundled_simulator =
        std::strcmp(name, "NetTLMSim-Fast") == 0 ||
        std::strcmp(name, "NetTLMSim-Full") == 0;
}

template <typename Function>
Function load_next_symbol(const char* name)
{
    void* symbol = dlsym(RTLD_NEXT, name);
    Function function = nullptr;
    static_assert(sizeof(function) == sizeof(symbol),
                  "Function and data pointers must have equal size");
    std::memcpy(&function, &symbol, sizeof(function));
    return function;
}

void resolve_symbols()
{
    real_new_array = load_next_symbol<new_array_fn>("_Znam");
    real_delete_array = load_next_symbol<delete_array_fn>("_ZdaPv");
    real_sized_delete_array =
        load_next_symbol<sized_delete_array_fn>("_ZdaPvm");
    detect_executable();

    if (real_new_array == nullptr || real_delete_array == nullptr) {
        write_message("NetTLM compatibility error: cannot resolve C++ array operators.\n");
        _exit(126);
    }
}

__attribute__((constructor))
void initialize_compatibility_layer()
{
    pthread_once(&symbols_once, resolve_symbols);
    if (is_bundled_simulator)
        write_message("NetTLM: SystemC stack compatibility layer loaded.\n");
}

bool track_mapping(void* address, std::size_t length)
{
    bool tracked = false;
    pthread_mutex_lock(&mapping_lock);
    if (mapping_count < max_tracked_stacks) {
        mappings[mapping_count++] = {address, length};
        tracked = true;
    }
    pthread_mutex_unlock(&mapping_lock);
    return tracked;
}

std::size_t consume_mapping(void* address)
{
    std::size_t length = 0;
    pthread_mutex_lock(&mapping_lock);
    for (std::size_t i = 0; i < mapping_count; ++i) {
        if (mappings[i].address == address) {
            length = mappings[i].length;
            mappings[i] = mappings[--mapping_count];
            break;
        }
    }
    pthread_mutex_unlock(&mapping_lock);
    return length;
}

void* allocate_large_array(std::size_t size)
{
    void* address = mmap(nullptr, size,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED)
        return nullptr;

    if (!track_mapping(address, size)) {
        (void)munmap(address, size);
        return nullptr;
    }

    if (!allocation_message_written) {
        allocation_message_written = true;
        write_message("NetTLM: using mmap-backed large allocations.\n");
    }
    return address;
}

} // namespace

void* operator new[](std::size_t size)
{
    pthread_once(&symbols_once, resolve_symbols);

    if (is_bundled_simulator && size >= mmap_allocation_threshold) {
        if (void* stack = allocate_large_array(size))
            return stack;
    }
    return real_new_array(size);
}

void operator delete[](void* address) noexcept
{
    if (address == nullptr)
        return;

    pthread_once(&symbols_once, resolve_symbols);
    const std::size_t length = consume_mapping(address);
    if (length != 0) {
        (void)munmap(address, length);
        return;
    }
    real_delete_array(address);
}

void operator delete[](void* address, std::size_t size) noexcept
{
    if (address == nullptr)
        return;

    pthread_once(&symbols_once, resolve_symbols);
    const std::size_t length = consume_mapping(address);
    if (length != 0) {
        (void)munmap(address, length);
        return;
    }

    if (real_sized_delete_array != nullptr)
        real_sized_delete_array(address, size);
    else
        real_delete_array(address);
}
