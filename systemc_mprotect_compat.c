#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * SystemC 2.3.3 QuickThreads can ask mprotect() to guard memory obtained from
 * new[].  On some RHEL/Rocky Linux glibc configurations that request fails
 * with EACCES.  The upstream fix treats stack protection as optional and only
 * restores a guard page if enabling it succeeded.
 *
 * The bundled simulators are statically linked to the affected SystemC, so the
 * implementation cannot be replaced without rebuilding them.  This preload
 * shim applies the same behavior at the mprotect boundary.  It only handles
 * failed PROT_NONE requests having SystemC's distinctive page_size - 1 length;
 * every other call and error is returned unchanged.
 */

typedef int (*mprotect_fn)(void *, size_t, int);

static mprotect_fn real_mprotect;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t skipped_lock = PTHREAD_MUTEX_INITIALIZER;
static void *skipped_guards[16384];
static size_t skipped_count;

static void write_message(const char *message)
{
    (void)write(STDERR_FILENO, message, strlen(message));
}

static void resolve_mprotect(void)
{
    *(void **)(&real_mprotect) = dlsym(RTLD_NEXT, "mprotect");
    if (real_mprotect == NULL)
        _exit(127);
}

/*
 * The allocator workaround is insufficient when SystemC successfully creates
 * a guard page but glibc later refuses to restore it.  Both bundled simulator
 * files are fixed-address, non-PIE ELF executables with symbols, so disable the
 * optional stack-protection method itself.  Validate its instruction prefix
 * before changing anything so this cannot silently patch a different build.
 */
static void disable_bundled_systemc_stack_protection(void)
{
    char executable[4096];
    const char *name;
    uintptr_t function_address = 0;
    static const unsigned char expected_prefix[] = { 0x55, 0x89, 0xf5, 0x53 };
    volatile unsigned char *function;
    long page_size;
    void *page;
    ssize_t length;

    length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length <= 0 || (size_t)length >= sizeof(executable) - 1)
        return;
    executable[length] = '\0';
    name = strrchr(executable, '/');
    name = name == NULL ? executable : name + 1;

    if (strcmp(name, "NetTLMSim-Fast") == 0)
        function_address = 0x4e3260u;
    else if (strcmp(name, "NetTLMSim-Full") == 0)
        function_address = 0x4e4fa0u;
    else
        return;

    function = (volatile unsigned char *)function_address;
    if (memcmp((const void *)function, expected_prefix, sizeof(expected_prefix)) != 0) {
        write_message("NetTLM compatibility error: unexpected SystemC binary version.\n");
        _exit(126);
    }

    pthread_once(&resolve_once, resolve_mprotect);
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        write_message("NetTLM compatibility error: cannot determine the page size.\n");
        _exit(126);
    }
    page = (void *)(function_address & ~((uintptr_t)page_size - 1u));

    if (real_mprotect(page, (size_t)page_size, PROT_READ | PROT_WRITE) != 0) {
        write_message("NetTLM compatibility error: cannot modify SystemC code page.\n");
        _exit(126);
    }
    function[0] = 0xc3; /* ret */
    __builtin___clear_cache((char *)function, (char *)function + 1);
    if (real_mprotect(page, (size_t)page_size, PROT_READ | PROT_EXEC) != 0) {
        write_message("NetTLM compatibility error: cannot restore SystemC code page.\n");
        _exit(126);
    }

    write_message("NetTLM: disabled incompatible SystemC 2.3.3 stack protection.\n");
}

__attribute__((constructor))
static void initialize_systemc_compatibility(void)
{
    disable_bundled_systemc_stack_protection();
}

static int is_systemc_guard_length(size_t length)
{
    long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 1 && length == (size_t)page_size - 1;
}

static int remember_skipped_guard(void *address)
{
    int remembered = 0;

    pthread_mutex_lock(&skipped_lock);
    if (skipped_count < sizeof(skipped_guards) / sizeof(skipped_guards[0])) {
        skipped_guards[skipped_count++] = address;
        remembered = 1;
    }
    pthread_mutex_unlock(&skipped_lock);

    return remembered;
}

static int consume_skipped_guard(void *address)
{
    size_t i;
    int found = 0;

    pthread_mutex_lock(&skipped_lock);
    for (i = 0; i < skipped_count; ++i) {
        if (skipped_guards[i] == address) {
            skipped_guards[i] = skipped_guards[--skipped_count];
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&skipped_lock);

    return found;
}

int mprotect(void *address, size_t length, int protection)
{
    int result;
    int saved_errno;

    pthread_once(&resolve_once, resolve_mprotect);

    if (is_systemc_guard_length(length) && protection != PROT_NONE &&
        consume_skipped_guard(address)) {
        errno = 0;
        return 0;
    }

    result = real_mprotect(address, length, protection);
    saved_errno = errno;

    if (result != 0 && saved_errno == EACCES && protection == PROT_NONE &&
        is_systemc_guard_length(length) && remember_skipped_guard(address)) {
        errno = 0;
        return 0;
    }

    errno = saved_errno;
    return result;
}
