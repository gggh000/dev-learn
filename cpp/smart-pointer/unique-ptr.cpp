#include <iostream>
#include <memory>
#include <utility>

using namespace std;
 
int main() {
    unique_ptr<int> valuePtr (new int (222) );
    cout << "*valuePtr: " << *valuePtr << ", valuePtr.get(): " << valuePtr.get() << endl ;
    unique_ptr<int> valuePtrNow(move(valuePtr));
    cout << "valuePtrNow: " << *valuePtrNow << ", valuePtrNow.get(): " << valuePtrNow.get() << endl;

    if (valuePtr.get() != NULL)
        cout << "*valuePtr: " << *valuePtr << ", valuePtr.get(): " << valuePtr.get() << endl ;
    else
        cout << "valuePtr.get(): " << valuePtr.get() << endl ;
    return 0;
}

