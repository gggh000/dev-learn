#include <iostream>
#include <memory>
#include <utility>

using namespace std;
 
int main() {
    shared_ptr<int> valuePtr (new int (222) );
    cout << "valuePtr: " << *valuePtr << endl;
    cout << "valuePtr.use_count() " << valuePtr.use_count() << endl;

    shared_ptr<int> valuePtrNow (valuePtr);
    shared_ptr<int> valuePtrNow2 (valuePtrNow);
    shared_ptr<int> valuePtrNow3 (valuePtr);

    cout << "valuePtr.get() " << valuePtr.get() << endl;
    cout << "valuePtrNow.get() " << valuePtrNow.get() << endl;

    cout << "valuePtr.use_count() " << valuePtr.use_count() << endl;
    cout << "valuePtrNow.use_count() " << valuePtrNow.use_count() << endl;
    
    return 0;
}

