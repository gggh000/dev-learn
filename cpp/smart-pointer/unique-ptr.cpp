#include <iostream>
#include <memory>
#include <utility>

using namespace std;
 
int main() {
    unique_ptr<int> valuePtr (new int (222) );
    cout << "valuePtr: " << *valuePtr << endl;
    unique_ptr<int> valuePtrNow(move(valuePtr));
    cout << "valuePtrNow: " << *valuePtrNow << endl;
    return 0;
}

