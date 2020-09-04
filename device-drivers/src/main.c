#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/stat.h>

MODULE_LICENSE("Dual BSD/GPL");

int myint=3;
module_param(myint, int, 0);

MODULE_PARM_DESC(myint, "An integer");
//module_param(myint, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

static int hello_init(void) {
    printk(KERN_ALERT "Hello, world.\n");
    printk(KERN_INFO "myint is an integer: %d\n", myint);
    return 0;
}

static void hello_exit(void) {
    printk(KERN_ALERT "Goodbye, cruel world.\n");
    return;
}

module_init(hello_init);
module_exit(hello_exit);
