AUTOMAKE_OPTIONS = subdir-objects 

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST	= example-cfg.json nomacro.pl

SUBDIRS		= compat

INCLUDES	= $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES)

bin_PROGRAMS	= minerd

dist_man_MANS	= minerd.1

minerd_SOURCES	= elist.h miner.h compat.h \
		  cpu-miner-work.c cpu-miner-opt.c cpu-miner.c util.c util-conversion.c \
		  sha2.c scrypt.c \
		  neoscrypt.c neoscrypt.h
if USE_ASM
minerd_SOURCES += Asm/neoscrypt_asm.S
if ARCH_x86
minerd_SOURCES += Asm/sha2-x86.S Asm/scrypt-x86.S 
endif
if ARCH_x86_64
minerd_SOURCES += Asm/sha2-x64.S Asm/scrypt-x64.S
endif
if ARCH_ARM
minerd_SOURCES += Asm/sha2-arm.S Asm/scrypt-arm.S
endif
endif
minerd_LDFLAGS	= $(PTHREAD_FLAGS)
minerd_LDADD	= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@
minerd_CPPFLAGS = @LIBCURL_CPPFLAGS@

