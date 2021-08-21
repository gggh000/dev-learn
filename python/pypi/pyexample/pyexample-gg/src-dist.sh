TEST_MODE=0
PACKAGE_NAME=pyexample-gg
pip3 install wheel
pip3 install twine
python3 setup.py sdist
python3 setup.py bdist_wheel --universal

if [[ $TEST_MODE -eq 1 ]] ; then
    echo "Test mode..."
    twine upload --repository-url https://test.pypi.org/legacy/ dist/$PACKAGE_NAME-0.1.0.tar.gz
else
    twine upload dist/*
fi 



