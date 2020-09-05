#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/fs.h>

#include "scull.h"

MODULE_LICENSE("Dual BSD/GPL");

int myint=3;
MODULE_PARM_DESC(myint, "An integer");
module_param(myint, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

int scull_major =   SCULL_MAJOR;
int scull_minor =   0;
int scull_nr_devs = SCULL_NR_DEVS;  /* number of bare scull devices */

static int hello_init(void) {
    printk(KERN_ALERT "Hello, world.\n");
    printk(KERN_INFO "myint is an integer: %d\n", myint);

    int result, i;
    dev_t dev = 0;

    /*if (scull_major) {
            dev = MKDEV(scull_major, scull_minor);
            result = register_chrdev_region(dev, scull_nr_devs, "scull");
    } else {
            scull_major = MAJOR(dev);
    }
    */

    result = alloc_chrdev_region(&dev, scull_minor, scull_nr_devs,  "scull");

    if (result < 0) {
            printk(KERN_WARNING "scull: can't get major %d\n", scull_major);
            return result;
}


    return 0;
}

static void hello_exit(void) {
    printk(KERN_ALERT "Goodbye, cruel world.\n");
    return;
}

module_init(hello_init);
module_exit(hello_exit);
