/*
TO build, use -std=c++0x switch with g++: 
i.e: g++ -std=c++0x  L21-functor.cpp -o a.out

Implementation of functor class

*/
#include <algorithm>
#include <iostream>
#include <vector>
#include <list>
using namespace std;

// struct that behaves as a unary function.

template <typename elementType>
struct displayElementKeepCount
{
    int count;

    displayElementKeepCount() 
    {
        count = 0;
    }

    elementType& operator=(const elementType & element)
    {
        vector <int> copy = new vector<int>;
        copy = element;
        return copy;
    }

    void operator()(const elementType & element)
    {
        ++count; 
        cout << count << ": " << element << ' ' << endl;
    }
};

int main()
{
    displayElementKeepCount<char>result; 
    vector <int> numsInVec{0, 1, 2, 3, 4};
    cout << "Vector of integers contains: " << endl;
    for_each(numsInVec.begin(), numsInVec.end(), displayElementKeepCount<int>() );
    
    // Display the list of characters.

    list <char> charsInList{'a', 'b', 'f', 't'};
    result = for_each (charsInList.begin(), charsInList.end(), displayElementKeepCount<char>());

    cout << "displayElementKeepCount was invoked " << result.count << " times." << endl;
    return 0;
}
