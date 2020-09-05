clear
echo "remove module..."
dkms remove  -m hello_dkms -v 1.0 --all

echo "copying module to /usr/src..."
rm -rf /usr/src/hello_dkms-1.0
mkdir /usr/src/hello_dkms-1.0
cd src
make clean
cd ..
cp -vr * /usr/src/hello_dkms-1.0/

echo "adding module..."
dkms add -m hello_dkms -v 1.0
echo "building module..."
dkms build  -m hello_dkms -v 1.0
echo "installing module..."
dkms install  -m hello_dkms -v 1.0

