#include <list>
#include <mutex>
#include <algorithm>
#include <thread> 
#include <iostream>

using namespace std;

std::mutex mutex_sum;
int sum = 0;

void hello(int pId, int pStat[])
{
    int pNum = rand() % 4 + 2;
    int sleep_duration = rand() % 30 + 20;
    std::cout << "(" << hex << this_thread::get_id() << ")" << pId << ": Hello CONCURRENT WORLD, sleeping for " << sleep_duration << endl;
    sleep(sleep_duration);

    // Create lock guard.

    std::lock_guard<std::mutex> sum_guard(mutex_sum);
    cout << "thread: " << pId << ": R0: " << sum;
    sum += pNum;
    cout << " + " << pNum << " = " << sum << endl;

    pStat[pId]  = 1;
    std::cout << pId << ": Done sleeping exiting now..." << endl;
}

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

    void add_using_mutex(int pThreadId, int pNum)
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
    // Declare, initialize variables.

    int i;
    const int CONFIG_THREAD_COUNT = 10;
    int stat[CONFIG_THREAD_COUNT];
    int sum = 0;

    // launch threads.

    for ( i = 0; i < CONFIG_THREAD_COUNT; i ++ ) {
        stat[i] = 0;
        std::thread t(hello, i, stat);
        t.detach();
    }

    cout << "Checking thread status-s..." << endl;

     while (sum != CONFIG_THREAD_COUNT)  {
        sum = 0;

        for (i = 0; i < CONFIG_THREAD_COUNT; i++) {
            cout << stat[i] << ", ";
            sum += stat[i];
        }

        cout << "main(): sum: " << sum << ". waiting for all threads to finish..." << endl;
        sleep(5);
    }

    return 0;
}
