#include <iostream>
#include <string>
using namespace std;
#define SIZE 40

class   static_ex {
public:
    static int static_member;
    int local_member;

    static_ex( int p_local_member ) : 
        local_member( p_local_member )  {
        static_member ++;
    }
};

int static_ex::static_member = 0;

int main()
{
    int i;
    static_ex * classArr[SIZE];
    int * intArr[SIZE];

    for ( i = 0 ; i < SIZE ; i ++ ) {
        classArr[i] = new static_ex ( i * SIZE * 11 );
        cout << endl << "------------------------";
        cout << endl << "((static_ex)classArr[i]).static_member: " << classArr[i]->static_member;
        cout << endl << "((static_ex)classArr[i]).local_member:  " << classArr[i]->local_member;
    }
    cout << endl << "------------------------";

    return 0;
}
