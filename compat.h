#ifndef __COMPAT_H__
#define __COMPAT_H__

#ifdef WIN32

#include <windows.h>
#include <time.h>

#ifndef localtime_r
#define localtime_r(src, dst) localtime_s(dst, src)
#endif

#define sleep(secs) Sleep((secs) * 1000)

enum {
	PRIO_PROCESS		= 0,
};

static __inline int setpriority(int which, int who, int prio)
{
	// Linux priority -19 - 19
	if (prio < 2) {
		prio = THREAD_PRIORITY_TIME_CRITICAL;
	} else if (prio < 1) {
		prio = THREAD_PRIORITY_ABOVE_NORMAL;
	} else if (prio > 2) {
		prio = THREAD_PRIORITY_IDLE;
	} else if (prio > 1) {
		prio = THREAD_PRIORITY_BELOW_NORMAL;
	}
	else
	{
		prio = THREAD_PRIORITY_NORMAL;
	}
	return -!SetThreadPriority(GetCurrentThread(), prio);
}

#ifdef _MSC_VER
#define snprintf(...) _snprintf(__VA_ARGS__)
#define strdup(...) _strdup(__VA_ARGS__)
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
#define __func__ __FUNCTION__
#define __thread __declspec(thread)
#define _ALIGN(x) __declspec(align(x))
typedef int ssize_t;

#include <stdlib.h>
// This static var is made to be compatible with linux/mingw (no free on string result)
// This is not thread safe but we only use that once on process start
static char dirname_buffer[_MAX_PATH] = { 0 };
static __inline char * dirname(char *file) {
	char drive[_MAX_DRIVE] = { 0 };
	char dir[_MAX_DIR] = { 0 };
	char fname[_MAX_FNAME], ext[_MAX_EXT];
	_splitpath_s(file, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);
	if (dir && strlen(dir) && dir[strlen(dir)-1] == '\\') {
		dir[strlen(dir) - 1] = '\0';
	}
	sprintf(dirname_buffer, "%s%s", drive, dir);
	return &dirname_buffer[0];
}
#endif

#endif /* WIN32 */

#ifndef _MSC_VER
#define _ALIGN(x) __attribute__ ((aligned(x)))
#endif

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef WIN32
#define MAX_PATH PATH_MAX
#endif

#endif /* __COMPAT_H__ */
