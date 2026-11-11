#pragma once

#include <functional>
#include <unordered_map>
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <typeindex>
#include <memory>
#include <thread>
#include <iostream>

class MessageBus {
public:
    // Publish một message kiểu T lên topic
    template<typename T>
    void publish(const std::string& topic, const T& msg) {
        get_or_create_queue<T>(topic)->publish(msg);
    }

    // Subscribe kiểu T trên topic, xử lý bằng callback
    template<typename T>
    void subscribe(const std::string& topic, std::function<void(const T&)> callback) {
        get_or_create_queue<T>(topic)->subscribe_loop(callback);
    }

private:
    struct IQueue {
        virtual ~IQueue() = default;
    };

    template<typename T>
    class MessageQueue : public IQueue {
    public:
        void publish(const T& msg) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(msg);
            }
            cond_.notify_all();
        }

        void subscribe_loop(std::function<void(const T&)> callback) {
            std::thread([this, callback]() {
                while (true) {
                    T msg;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cond_.wait(lock, [&] { return !queue_.empty(); });
                        msg = queue_.front();
                        queue_.pop();
                    }
                    callback(msg);
                }
            }).detach();
        }

    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };

    template<typename T>
    std::shared_ptr<MessageQueue<T>> get_or_create_queue(const std::string& topic) {
        std::lock_guard<std::mutex> lock(bus_mutex_);

        auto it = queues_.find(topic);
        if (it != queues_.end()) {
            auto expected = types_.find(topic);
            if (expected != types_.end() && expected->second != std::type_index(typeid(T))) {
                throw std::runtime_error("Type mismatch on topic: " + topic);
            }

            return std::static_pointer_cast<MessageQueue<T>>(it->second);
        }

        std::type_index ti(typeid(T));
        auto q = std::make_shared<MessageQueue<T>>();

        queues_.insert({topic, q});
        types_.insert({topic, ti});

        return q;
    }


    std::mutex bus_mutex_;
    std::unordered_map<std::string, std::shared_ptr<IQueue>> queues_;
    std::unordered_map<std::string, std::type_index> types_;
};
