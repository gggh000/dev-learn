///usr/src/kernels/3.10.0-327.el7.x86_64/arch/x86/include/asm/init.h
///usr/src/kernels/3.10.0-327.el7.x86_64/include/config/debug/memory/init.h
///usr/src/kernels/3.10.0-327.el7.x86_64/include/config/provide/ohci1394/dma/init.h
///usr/src/kernels/3.10.0-327.el7.x86_64/include/linux/init.h

///usr/src/kernels/3.10.0-327.el7.x86_64/arch/x86/include/asm/linkage.h
///usr/src/kernels/3.10.0-327.el7.x86_64/include/asm-generic/linkage.h
///usr/src/kernels/3.10.0-327.el7.x86_64/include/linux/linkage.h
///usr/src/kernels/3.10.0-327.el7.x86_64.debug/arch/x86/include/asm/linkage.h
///usr/src/kernels/3.10.0-327.el7.x86_64.debug/include/asm-generic/linkage.h
///usr/src/kernels/3.10.0-327.el7.x86_64.debug/include/linux/linkage.h

#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("Dual BSD/GPL")

static int hello_init(void) {
    printk(KERN_ALERT "Hello, world.\n");
    return 0;
}

static void hello_exit(void) {
    printk(KERN_ALERT "Goodbye, cruel world.\n");
}

module_init(hello_init);
module_exit(hello_exit);
