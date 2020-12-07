# detach-disk not working while vm running due to hotplug disk is not supported for this type of disk
# https://serverfault.com/questions/457250/kvm-and-libvirt-how-do-i-hotplug-a-new-virtio-disk
# root@ixt-hq-44:/git.co/dev-learn/c/asm-c-linkage# ./config-target-vm.sh rm
# removing second hdd
# error: Failed to detach disk
# error: Operation not supported: This type of disk cannot be hot unplugged

P1=$1
VM_NAME=minix-boot
IMAGE_NAME=/var/lib/libvirt/images/minix-boot-1.qcow2

if [[ $P1 == "rm" ]] ; then
	echo "removing second hdd"
	virsh detach-disk --domain $VM_NAME $IMAGE_NAME --persistent --config --live
	virsh domblklist $VM_NAME
elif [[ $P1 == "add" ]] ; then
	echo "attaching second hdd"
	virsh attach-disk --domain $VM_NAME --source $IMAGE_NAME --target hdc --config --live
	virsh domblklist $VM_NAME
else
	echo "invalid parameter: $P1 "
fi
