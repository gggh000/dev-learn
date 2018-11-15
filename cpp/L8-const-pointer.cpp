#include <iostream>
using namespace std;

main()
{
    cout << "Data can be changed but not pointer example: " << endl;

    int data1 = 100;
    int data2 = 200;
    int * const p1 = &data1;

    // Uncommenting following will cause compile error since pointer pl is constant.
    //p1 = &data2;

    cout << "Data can not be changed but pointer can be, example: " << endl;

    int data3 = 300;
    int data4 = 400;

    const int * p2 = &data3;
    p2 = &data4;

    // Uncommenting following will cause compile error because it is pointer to const int.
    //*p2 = 500;    

    return 0;
}
