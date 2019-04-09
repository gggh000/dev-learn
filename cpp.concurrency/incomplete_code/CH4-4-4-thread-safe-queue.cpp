#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class threadsafe_queue {
private:

    /* Declare mutex, queue and condition variable as members. */ 
    std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;

public:

    // Thread-safe push by acquiring lock first on mutex.

    void push( T new value ) {
        std::lock_guard<std::mutex> lk ( mut );
        data_queue.push(new_value);
        data_cond.notify_one();
    }

    void wait_and_pop( T & value) {
        std::unique_lcok<std::mutex> lk ( mut );
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        value = data_queue.front();
        data_queue.pop();
    }
};

threadsafe_queue<data_chunk> data_queue; 

void data_preparation_thread() {
    while ( more_data_to_prepare()) {
        data_chunk const data=prepare_data();       
        data_queue.push(data);
    }
}

void data_processing_thread() {
    while ( true ) {
        data_chunk data;
        data_queue.wait_and_pop(data);
        process(data);  
        
        if (is_last_chunk(data))
            break;
    }
}




