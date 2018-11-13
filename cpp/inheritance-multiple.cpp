/*
Example of multiple inheritance where class Platypus is inherited from 3 classes Mammal,
Reptile and Bird including its methods. Two of the classes, Reptile and Bird has 
same function spitVenom which compiler complains unless explicitly called with
scope resolution operator ::.
*/
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
    void spitVenom()
    {
        cout << "Bird: Spits venom?" << endl;
    }
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
    realFreak.Bird::spitVenom();
    realFreak.Reptile::spitVenom();
    realFreak.Swim();
}
