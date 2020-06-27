#ifndef __glutwin32_h__
#define __glutwin32_h__

/* Copyright (c) Nate Robins, 1997. */
/* portions Copyright (c) Mark Kilgard, 2000. */

/* This program is freely distributable without licensing fees 
   and is provided without guarantee or warrantee expressed or 
   implied. This program is -not- in the public domain. */

#include "win32_x11.h"
#include "win32_glx.h"

/* We have to undef some things because Microsoft likes to pollute the
   global namespace. */
#undef TRANSPARENT

/* Polymorphic data type not present in older (32 bit only) header files. */
#if _MSC_VER < 1200
typedef unsigned int UINT_PTR;
#endif

/* Win32 "equivalent" cursors - eventually, the X glyphs should be
   converted to Win32 cursors -- then they will look the same */
#define XC_arrow               IDC_ARROW
#define XC_top_left_arrow      IDC_ARROW
#define XC_hand1               IDC_SIZEALL
#define XC_pirate              IDC_NO
#define XC_question_arrow      IDC_HELP
#define XC_exchange            IDC_NO
#define XC_spraycan            IDC_SIZEALL
#define XC_watch               IDC_WAIT
#define XC_xterm               IDC_IBEAM
#define XC_crosshair           IDC_CROSS
#define XC_sb_v_double_arrow   IDC_SIZENS
#define XC_sb_h_double_arrow   IDC_SIZEWE
#define XC_top_side            IDC_UPARROW
#define XC_bottom_side         IDC_SIZENS
#define XC_left_side           IDC_SIZEWE
#define XC_right_side          IDC_SIZEWE
#define XC_top_left_corner     IDC_SIZENWSE
#define XC_top_right_corner    IDC_SIZENESW
#define XC_bottom_right_corner IDC_SIZENWSE
#define XC_bottom_left_corner  IDC_SIZENESW

#define XA_STRING 0

/* Use our own typedef for timeval to avoid conflicts with the
   struct timeval defined by the Cygwin32 and/or MingW32 headers. */
struct timevalWIN32 {
  long tv_sec;     /* seconds */
  long tv_usec;    /* microseconds */
};

/* Private routines from win32_util.c */
extern int gettimeofdayWIN32(struct timevalWIN32* tp);
extern void *__glutFont(void *font);
extern int __glutGetTransparentPixel(Display *dpy, XVisualInfo *vinfo);
extern void __glutAdjustCoords(Window parent, int *x, int *y, int *width, int *height);

#endif /* __glutwin32_h__ */
