#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/slab.h> // for qset, quantum
#include <linux/cdev.h> // for cdev

#include "scull.h"

MODULE_LICENSE("Dual BSD/GPL");

int myint=3;
MODULE_PARM_DESC(myint, "An integer");
module_param(myint, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

int scull_major =   SCULL_MAJOR;
int scull_minor =   0;
int scull_nr_devs = SCULL_NR_DEVS;  /* number of bare scull devices */
int scull_quantum = SCULL_QUANTUM;
int scull_qset =    SCULL_QSET;

int scull_release(struct inode * inode, struct file *filp) {
    return 0;
}
int scull_trim(struct scull_dev *dev)
{
    struct scull_qset *next, *dptr;
    int qset = dev->qset;   /* "dev" is not-null */
    int i;

    for (dptr = dev->data; dptr; dptr = next) { /* all the list items */
        if (dptr->data) {
            for (i = 0; i < qset; i++)
                kfree(dptr->data[i]);
            kfree(dptr->data);
            dptr->data = NULL;
        }
        next = dptr->next;
        kfree(dptr);
    }
    dev->size = 0;
    dev->quantum = scull_quantum;
    dev->qset = scull_qset;
    dev->data = NULL;
    return 0;
}

int scull_open(struct inode  * inode, struct file * filp) {
    struct scull_dev * dev;
    dev = container_of(inode->i_cdev, struct scull_dev, cdev);
    filp->private_data = dev;
    
    if (( filp->f_flags & O_ACCMODE ) == O_WRONLY) {
        scull_trim(dev);
    }
    return 0;
}


struct file_operations scull_fops = {
    .owner =    THIS_MODULE,
    //.llseek =   scull_llseek,
    //.read =     scull_read,
    //.write =    scull_write,
    //.unlocked_ioctl = scull_ioctl,
    .open =     scull_open,
    .release =  scull_release,
};

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
    } else {
        printk(KERN_INFO "acquired major number: %d\n", result);
    }

    return 0;
}

static void hello_exit(void) {
    printk(KERN_ALERT "Goodbye, cruel world.\n");
    return;
}

module_init(hello_init);
module_exit(hello_exit);


