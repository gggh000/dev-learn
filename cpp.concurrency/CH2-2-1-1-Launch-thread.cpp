#include <iostream>
#include <thread>
using namespace std;

class background_task
{
public:
    void do_something() const { 
        cout << "do_something." << endl; 
    }

    void do_something_else() const { 
        cout << "do_something_else." << endl; 
    }

    void operator () () const
    {
        do_something();
        do_something_else();
    }
};

int main()
{
    background_task f;
    std::thread my_thread(f);
    my_thread.join();
}
