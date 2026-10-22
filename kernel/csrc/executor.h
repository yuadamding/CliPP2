// One fixed native worker team. Ordered outputs; disjoint tasks own arithmetic.
#pragma once
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace clipp2kernel {
class Executor {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable ready, finished;
    std::atomic<int> next{0};
    int count=0, pending=0;
    unsigned long epoch=0;
    bool stopping=false;
    bool each_worker=false;
    void (*operation)(void*, int)=nullptr;
    void* context=nullptr;
    std::exception_ptr failure;
    void consume();
    void worker();
public:
    explicit Executor(int workers);
    ~Executor();
    Executor(const Executor&)=delete;
    Executor& operator=(const Executor&)=delete;
    template<class Function> void run(int tasks, Function&& function, bool once_per_thread=false) {
        if (threads.empty()) { for(int i=0;i<tasks;++i) function(i); return; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            context=&function;
            operation=[](void* data,int i){(*static_cast<typename std::remove_reference<Function>::type*>(data))(i);};
            count=tasks;next=0;failure=nullptr;each_worker=once_per_thread;pending=threads.size();++epoch;
        }
        ready.notify_all();consume();
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock,[&]{return pending==0;});
        if(failure)std::rethrow_exception(failure);
    }
    template<class Function> void each(Function&& function) {run(1,function,true);}
};
}
