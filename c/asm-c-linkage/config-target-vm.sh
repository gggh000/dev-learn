P1=$1
VM_NAME=minix-boot
IMAGE_NAME=/var/lib/libvirt/images/minix-boot-1.qcow2

if [[ $P1 == "rm" ]] ; then
	echo "removing second hdd"
	virsh detach-disk --domain $VM_NAME $IMAGE_NAME --persistent --config
	virsh domblklist $VM_NAME
elif [[ $P1 == "add" ]] ; then
	echo "attaching second hdd"
	virsh attach-disk --domain $VM_NAME --source $IMAGE_NAME --target hdc --config
	virsh domblklist $VM_NAME
else
	echo "invalid parameter: $P1 "
fi
