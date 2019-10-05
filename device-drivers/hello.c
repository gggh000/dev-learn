#ifndef _LINUX_INIT_H
#define _LINUX_INIT_H

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

//MODULE_LICENSE("Dual BSD/GPL")

static int __init helloworld_init(void) {
//  printk(KERN_ALERT "Hello, world.\n");
//    printk("Hello, world.\n");
	pr_info("Hello world.\n");
    return 0;
}

static void __exit helloworld_exit(void) {
//  printk(KERN_ALERT "Goodbye, cruel world.\n");
//    printk("Goodbye, cruel world.\n");	
	pr_info("Bye world\n");
}

module_init(helloworld_init);
module_exit(helloworld_exit);

#endif
