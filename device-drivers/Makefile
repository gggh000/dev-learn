CC=gcc

I1=/usr/src/linux-headers-5.0.0-29/include/
I2=/usr/src/linux-headers-5.0.0-29/arch/x86/include/
#I1=/usr/src/kernels/3.10.0-327.el7.x86_64/arch/x86/include/
#I2=/usr/src/kernels/3.10.0-327.el7.x86_64/include/
#I3=/usr/src/kernels/2.6.32-754.10.1.el6.x86_64.debug/include/

CFLAGS=-I $(I1) -I $(I2)
#CFLAGS=-O3

obj-m := hello.o
hello-objs := hello.o

#hello1: hello.c
#	$(CC) $(CFLAGS) hello.c


#obj-m := gim.o
#gim-objs := main.o 



PWD := $(shell pwd)

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
#KERNELDIR ?= /lib/modules/3.4.9/build
#KERNELDIR ?= /lib/modules/3.4.9-test+/build
#KERNELDIR ?= /work_test/git/kvm
#KERNELDIR ?= ../../linux

#$(info $(KBUILD_CFLAGS))
all:
	$(MAKE) -C $(KERNELDIR) 

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
