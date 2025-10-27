/**
 * \file gft_common.h
 * \brief Header file for common definitions and function prototypes.
 */

#ifndef _GFT_COMMON_H_
#define _GFT_COMMON_H_

/* C headers */
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <time.h>

/* OpenMP header included normally */
#include <omp.h>

/* SIMD intrinsics */
#include <xmmintrin.h>

#ifdef _WIN32
#include <windows.h>
/* timeval may be provided by some Windows headers; only define if missing */
#if !defined(_TIMEVAL_DEFINED) && !defined(_WINSOCKAPI_) && !defined(_WINSOCK2API_)
struct timeval {
  long tv_sec;
  long tv_usec;
};
#define _TIMEVAL_DEFINED
#endif

static inline int gettimeofday(struct timeval *tv, void *tz)
{
  FILETIME ft;
  unsigned long long tmpres = 0;
  static const unsigned long long EPOCH_DIFF = 11644473600000000ULL; // 1970-01-01

  if (tv == NULL)
    return -1;

  GetSystemTimeAsFileTime(&ft);

  tmpres |= ft.dwHighDateTime;
  tmpres <<= 32;
  tmpres |= ft.dwLowDateTime;

  /* convert into microseconds */
  tmpres /= 10;
  tmpres -= EPOCH_DIFF;

  tv->tv_sec = (long)(tmpres / 1000000ULL);
  tv->tv_usec = (long)(tmpres % 1000000ULL);

  return 0;
}
#else
#include <sys/time.h>
#endif

/**
 * \brief Base namespace for common definitions and prototypes.
 */
namespace gft{

/* check all errors explicitly */
//#define GFT_CHECK_ALL_ERRORS  1

/* Error messages */
#define MSG1  "Cannot allocate memory space"
#define MSG2  "Cannot open file"
#define MSG3  "Invalid option"
#define MSG4  "Could not locate nifti header"
#define MSG5  "Nifti-1 data type not supported"

/* Common data types to all programs */
#ifndef __cplusplus
typedef enum boolean {false,true} bool;
#endif
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned char uchar;

typedef struct timeval timer;


/* Common definitions */
#define PI          3.1415926536
#define INTERIOR    0
#define EXTERIOR    1
#define BOTH        2
#define WHITE       0
#define GRAY        1
#define BLACK       2
#define NIL        -1
#define INCREASING  1
#define DECREASING  0
#define Epsilon     1E-05

/* Common operations */

  /**
   * \def MAX(x,y)
   * \brief A macro that returns the maximum of \a x and \a y.
   */
#ifndef MAX
#define MAX(x,y) (((x) > (y))?(x):(y))
#endif
  /**
   * \def MIN(x,y)
   * \brief A macro that returns the minimum of \a x and \a y.
   */
#ifndef MIN
#define MIN(x,y) (((x) < (y))?(x):(y))
#endif

#define ROUND(x) ((x < 0)?(int)(x-0.5):(int)(x+0.5))

#define SIGN(x) ((x >= 0)?1:-1)

#define SQUARE(x) ((x)*(x))

#if defined(_MSC_VER)
  /* MSVC: provide simple vector-like structs with basic operators used in code */
  struct v4sf { float d[4]; inline v4sf operator+(const v4sf &o) const { v4sf r; for(int i=0;i<4;i++) r.d[i]=d[i]+o.d[i]; return r; } inline v4sf operator-(const v4sf &o) const { v4sf r; for(int i=0;i<4;i++) r.d[i]=d[i]-o.d[i]; return r; } inline v4sf operator-() const { v4sf r; for(int i=0;i<4;i++) r.d[i]=-d[i]; return r; } };
  struct v4si { int   d[4]; inline v4si operator+(const v4si &o) const { v4si r; for(int i=0;i<4;i++) r.d[i]=d[i]+o.d[i]; return r; } inline v4si operator-(const v4si &o) const { v4si r; for(int i=0;i<4;i++) r.d[i]=d[i]-o.d[i]; return r; } inline v4si operator-() const { v4si r; for(int i=0;i<4;i++) r.d[i]=-d[i]; return r; } inline v4si& operator*=(const v4si &o){ for(int i=0;i<4;i++) d[i]*=o.d[i]; return *this; } inline v4si& operator*=(int v){ for(int i=0;i<4;i++) d[i]*=v; return *this; } };
  struct v8qi { unsigned char d[8]; inline v8qi operator+(const v8qi &o) const { v8qi r; for(int i=0;i<8;i++) r.d[i]=d[i]+o.d[i]; return r; } inline v8qi operator-(const v8qi &o) const { v8qi r; for(int i=0;i<8;i++) r.d[i]=d[i]-o.d[i]; return r; } };
  struct v16qi{ unsigned char d[16]; inline v16qi operator+(const v16qi &o) const { v16qi r; for(int i=0;i<16;i++) r.d[i]=d[i]+o.d[i]; return r; } inline v16qi operator-(const v16qi &o) const { v16qi r; for(int i=0;i<16;i++) r.d[i]=d[i]-o.d[i]; return r; } };
  struct v8hi { unsigned short d[8]; inline v8hi operator+(const v8hi &o) const { v8hi r; for(int i=0;i<8;i++) r.d[i]=d[i]+o.d[i]; return r; } inline v8hi operator-(const v8hi &o) const { v8hi r; for(int i=0;i<8;i++) r.d[i]=d[i]-o.d[i]; return r; } };

  /* Provide element-wise multiplication assignment for MSVC emulated vector types */
  struct v8qi;
  struct v16qi;
  struct v8hi;
  
  /* define *= for unsigned char vectors */
  inline v8qi &operator*=(v8qi &a, const v8qi &b){ for(int i=0;i<8;i++) a.d[i] = (unsigned char)(a.d[i] * b.d[i]); return a; }
  inline v16qi &operator*=(v16qi &a, const v16qi &b){ for(int i=0;i<16;i++) a.d[i] = (unsigned char)(a.d[i] * b.d[i]); return a; }
  /* define *= where right operand is v4si (int vector), convert elementwise */
  inline v8hi &operator*=(v8hi &a, const v8hi &b){ for(int i=0;i<8;i++) a.d[i] = (unsigned short)(a.d[i] * b.d[i]); return a; }
  /* scalar variants */
  inline v8qi &operator*=(v8qi &a, unsigned char v){ for(int i=0;i<8;i++) a.d[i] = (unsigned char)(a.d[i] * v); return a; }
  inline v16qi &operator*=(v16qi &a, unsigned char v){ for(int i=0;i<16;i++) a.d[i] = (unsigned char)(a.d[i] * v); return a; }
  inline v8hi &operator*=(v8hi &a, unsigned short v){ for(int i=0;i<8;i++) a.d[i] = (unsigned short)(a.d[i] * v); return a; }
  inline v4sf &operator*=(v4sf &a, float v){ for(int i=0;i<4;i++) a.d[i] = a.d[i] * v; return a; }
  inline v4sf &operator*=(v4sf &a, const v4sf &b){ for(int i=0;i<4;i++) a.d[i] = a.d[i] * b.d[i]; return a; }
  /* v4si provides member operator*=; avoid duplicate free-function overloads to prevent ambiguity */
#endif
#if defined(_WIN32)
  /* Provide small compatibility wrappers for POSIX functions used in the code */
  #include <string.h>
  #include <stdlib.h>
  /* Provide alternative operator tokens (and, or, not) on MSVC */
  #include <iso646.h>
  static inline void bzero(void *s, size_t n){ memset(s,0,n); }
  #if !defined(strcasecmp)
    #define strcasecmp _stricmp
  #endif
#else
  /**
   * \brief Vector of four single floats.
   */
  typedef float  v4sf  __attribute__ ((vector_size(16),aligned(16)));
  
  /**
   * \brief Vector of four single integers.
   */
  typedef int    v4si  __attribute__ ((vector_size(16),aligned(16)));

  /**
   * \brief Vector of eight unsigned 8-bit integers.
   */
  typedef uchar  v8qi  __attribute__ ((vector_size(8),aligned(16)));

  /**
   * \brief Vector of sixteen unsigned 8-bit integers.
   */
  typedef uchar  v16qi __attribute__ ((vector_size(16),aligned(16)));

  /**
   * \brief Vector of eight unsigned short integers.
   */
  typedef ushort v8hi  __attribute__ ((vector_size(16),aligned(16)));
#endif
  
  
  typedef union _voxel {
    v4si v;
    int  data[4];
    /* expose x,y,z directly so code can use voxel.x / voxel.y / voxel.z */
    int x; int y; int z; int _pad_v;
    struct{ int x,y,z; } c;
  } Voxel;


  typedef struct _pixel {
    int x,y;
  } Pixel;
  

  char   *AllocCharArray(int n);  /* It allocates 1D array of n characters */


  /**
   * \brief It allocates 1D array of n characters.
   */
  uchar  *AllocUCharArray(int n);

  /**
   * \brief It allocates 1D array of n ushorts.
   */
  ushort *AllocUShortArray(int n);

  uint   *AllocUIntArray(int n); /* It allocates 1D array of n uints */
  
  /**
   * \brief It allocates 1D array of n integers.
   */
  int    *AllocIntArray(int n);

  /**
   * \brief It allocates 1D array of n long long integers.
   */
  long long  *AllocLongLongArray(int n);  
  
  /**
   * \brief It allocates 1D array of n floats.
   */
  float  *AllocFloatArray(int n);

  double *AllocDoubleArray(int n);/* It allocates 1D array of n doubles */

  void    FreeIntArray(int **a);
  void    FreeFloatArray(float **a);
  void    FreeDoubleArray(double **a);
  void    FreeUCharArray(uchar **a);
  void    FreeUShortArray(ushort **a);

  /**
   * \brief It prints error message and exits the program.
   */
  void Error(char *msg,char *func);

  /**
   * \brief It prints warning message and leaves the routine.
   */
  void Warning(char *msg,char *func);

  /**
   * \brief It changes content between a and b.
   */
  inline void SwapInt(int *a, int *b){
    int c;
    c  = *a;  
    *a = *b;  
    *b = c;
  }

  /**
   * \brief It changes content between a and b.
   */
  inline void SwapFloat(float *a, float *b){
    float c;
    c  = *a;
    *a = *b;
    *b = c;
  }

  
  int NCFgets(char *s, int m, FILE *f); /* It skips # comments */
  
  /**
   * Gera um número inteiro aleatório no intervalo [low,high].
   http://www.ime.usp.br/~pf/algoritmos/aulas/random.html
   Execute RandomSeed antes, para atualizar a semente.
  */
  int  RandomInteger (int low, int high);
  void RandomSeed();

  /*
  int   IntegerNormalize(int value,
			 int omin,int omax,
			 int nmin,int nmax);
  */

  int   LinearStretch(int value,
		      int omin, int omax,
		      int nmin, int nmax);
  float LinearStretch(float value,
		      float omin, float omax,
		      float nmin, float nmax);
  double LinearStretch(double value,
		       double omin, double omax,
		       double nmin, double nmax);
  
} //end gft namespace

#endif

