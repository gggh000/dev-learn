CC=gcc
I1=/usr/src/kernels/3.10.0-327.el7.x86_64/arch/x86/include/
I2=/usr/src/kernels/3.10.0-327.el7.x86_64/include/
I3=/usr/src/kernels/2.6.32-754.10.1.el6.x86_64.debug/include/

CFLAGS=-I $(I1) -I $(I2)
#CFLAGS=-O3

hello1: hello.c
	$(CC) $(CFLAGS) hello.c
