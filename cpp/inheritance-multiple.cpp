#include <iostream>
using namespace std;

class Mammal
{
public:
    void feedMyMilk()
    {
        cout << "Mammal: Baby says glug!" << endl;
    }
};

class Reptile 
{
public:
    void spitVenom()
    {
        cout << "Reptile: Spits venom!" << endl;
    }
};

class Bird 
{
public:
    void LayEggs()
    {
        cout << "Bird: lay eggs!" << endl;
    }
};

class Platypus : public Bird, public Reptile, public Mammal
{
public:
    void Swim()
    {
        cout << "Platypus: Voila, I can swim!" << endl;
    }
};

int main()
{
    Platypus realFreak;
    realFreak.LayEggs();
    realFreak.feedMyMilk();
    realFreak.spitVenom();
    realFreak.Swim();
}
