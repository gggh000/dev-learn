/*Simple thread app which launches hello and waits.
Use: g++ -std=gnu++0x -pthread hello.cpp to build.
*/

#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <exception>

using namespace std;

void hello(int pId, int pStat[])
{
    int sleep_duration = rand() % 30 + 20;
	std::cout << "(" << hex << this_thread::get_id() << ")" << pId << ": Hello CONCURRENT WORLD, sleeping for " << sleep_duration << endl; 
    sleep(sleep_duration);
    pStat[pId]  = 1;
    std::cout << pId << ": Done sleeping exiting now..." << endl;
}

int main()
{
    int i;
    const int CONFIG_THREAD_COUNT = 10;
    int stat[CONFIG_THREAD_COUNT];
    int sum = 0;

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
}
