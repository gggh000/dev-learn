/*
Waiting for data to be proessed with a std::condition_variable
Chapter 4
Listing 4.1
*/


std::mutex mut;
std::queue<data_chunk> data_queue;
std::condition_variable data_cond;

/* Function used for preparing / pushing data onto queue. 
During push, it protects data queue using mutex by wrapping in lock_guard 
which uses RAII */

void data_preparation_thread() {
    while (more_data_to_prepare()) {
        data_chunk const data = prepare_data()
        std::lock_guard<std::mutex> lk ( mut );
        data_queue.push(data);
        data_cond.notify_one();
    }
}

/*  While data is fed through push by data_preparation_thread, the data_processing_thread will consume the data 
Since data_preparation_thread locks the queue while pushing data_processing_thread() must wait using 
same data_cond variable by instantiating the mutext with lock. 
Once wait is over the queue.front is assigned to data of data_chunk type 
and popped from queue. Once the data is popped from front of queue lock is released. Since lock is a type 
of unique_lock not std_lock it is not using RAII idiom which amounts to more flexibility. 
*/ 

void data_processing_thread() {
    while ( true ) {
        std::unique_lock<std::mutex> lk ( mut );
        data_cond.wait( lk, [] { return !data_queue.empty() ; })  ;
        data_chunk data = data_queue.front();
        data_queue.pop()
        lk.unlock ();
        process( data );
        
        if ( is_last_chunk ( data ))
            break;

    }
}
