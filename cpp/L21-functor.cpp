#include <algorithm>
#include <iostream>
#include <vector>
#include <list>
using namespace std;

// struct that behaves as a unary function.

template <typename elementType>
struct displayElement
{
    void operator () (const elementType & element) const
    {
        cout << element << ' ';
    }
};

int main()
{
    vector <int> numsInVec{0, 1};
    cout << "Vector of integers contains: " << endl;
    for_each (numsInVec.begin(), numsInVec.end(), displayElement<int>() );
    
    // Display the list of characters.

    list <char> charsInList{'a', 'b', 'f', 't'};
    for_each (charsInList.begin(), chatsInList.end(), displayElement<char>());

    return 0;
}
