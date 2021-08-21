mkdir -p hello_1.0-1_amd64/usr/local/bin
#mkdir -p helloworld_1.0-1_amd64/DEBIAN
#touch helloworld_1.0-1_amd64/DEBIAN/control

cp ./proj1/hello ./hello_1.0-1_amd64/usr/local/bin/
cp ./proj1/hello.py ./hello_1.0-1_amd64/usr/local/bin/
dpkg-deb --build --root-owner-group hello_1.0-1_amd64

