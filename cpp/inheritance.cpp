/*
Simple example of public inheritance.
*/
#include <iostream>
using namespace std;

class Fish 
{
public:
    bool isFreshWaterFish;

    void Swim()
    {
        if (isFreshWaterFish) 
            cout << "Can swim in lake." << endl;
        else
            cout << "Can swim in sea." << endl;
    }
};

class Tuna : public Fish
{
public:
    Tuna()
    {
        isFreshWaterFish = false;
    }
};

class Carp : public Fish
{
public:
    Carp()
    {
        isFreshWaterFish = true;
    }
};

int main()
{
    Carp myLunch;
    Tuna myDinner;
    cout << "About my food" << endl;

    cout << "Lunch: " ;
    myLunch.Swim();
    cout << "Dinner: " ;
    myDinner.Swim();
}
