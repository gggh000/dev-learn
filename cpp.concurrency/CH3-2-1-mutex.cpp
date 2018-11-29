#include <list>
#include <mutex>
#include <algorithm>
#include <thread> 

using namespace std;

class data_wrapper
{
private:
    int sum;      
    std::mutex m;

public:
    data_wrapper()
    {
        cout << "data_wrapper constructor." << endl;
        sum = 0;
    }

    add_using_mutex(int pThreadId, pNum)
    {
        std::lock_guard<std::mutex> guard(m);

        // read current value.
        // add random value.
        // read back sum.
        // (if no race condition, R + ADD + R should be consistent.

        cout << "thread: " << pThreadId << ": R0: " << sum;
        sum += pNum;
        cout << " + " << pNum << " = " << sum << endl;
    }
}; 

int main()
{

    return 0;
}
