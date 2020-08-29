CC=g++

#I1=/usr/src/linux-headers-5.0.0-29/include/
#I2=/usr/src/linux-headers-5.0.0-29/arch/x86/include/

#CFLAGS=-I $(I1) -I $(I2)
#CFLAGS=-O3

obj-m := hello.o

#PWD := $(shell pwd)

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
all:
	make -C $(KERNELDIR)  M=$(PWD) modules

#clean:
#	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
